"""Exercise the Inkling development API from inside the engine container.

Adapted from kimi-k31-cc-lab/scripts/qualify-api.py (the 16-case gate).
Run via docker exec against vLLM's loopback port; timings are not benchmarks.
"""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from openai import DefaultHttpxClient, OpenAI

MODEL = 'inkling-2-small'
OUTPUT_LIMIT = 4096
TIMEOUT_SECONDS = 300
SEED = 20261113
REASONING_EFFORTS = ('low', 'high', 'max')
TOOL_NAME = 'lookup_inventory'
TOOL_ARGUMENTS = {'sku': 'INK2-LAB'}
TOOL_RESULT = {'available': 17}
TOOL_ANSWER = 'Inventory: 17'
TOOLS = [{'type': 'function', 'function': {'name': TOOL_NAME,
          'description': 'Look up inventory for a SKU.',
          'parameters': {'type': 'object', 'properties': {'sku': {'type': 'string'}},
                         'required': ['sku'], 'additionalProperties': False}}}]
PREFIX_ROWS = 4096
CANCEL_EVENT_COUNT = 3
CANCEL_TOKEN_LIMIT = 8192


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def schema(expected):
    properties = {key: {'type': 'integer' if type(value) is int else 'string'}
                  for key, value in expected.items()}
    return {'type': 'json_schema', 'json_schema': {'name': 'lab_result', 'strict': True,
            'schema': {'type': 'object', 'properties': properties,
                       'required': list(expected), 'additionalProperties': False}}}


def payload(prompt, *, stream=False, effort=None):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': SEED, 'max_tokens': OUTPUT_LIMIT, 'stream': stream}
    if effort is not None:
        body['reasoning_effort'] = effort
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Loopback origin, without /v1')
    parser.add_argument('--fixtures', type=Path,
                        default=Path(__file__).resolve().parent / 'fixtures')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Run against the engine loopback only')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'model': MODEL,
              'scope': 'Development API correctness; timings are not benchmarks',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': []}
    client = OpenAI(base_url=args.base_url.rstrip('/') + '/v1', api_key='loopback',
                    timeout=TIMEOUT_SECONDS, max_retries=0,
                    http_client=DefaultHttpxClient(trust_env=False, timeout=TIMEOUT_SECONDS))

    def request(name, body, *, expected=None, json_expected=None, tool=False):
        save(args.output / (name + '.request.json'), body)
        row = {'name': name, 'passed': False}
        report['cases'].append(row)
        started = time.monotonic()
        try:
            if body['stream']:
                stream_args = {key: value for key, value in body.items() if key != 'stream'}
                stream_args['stream_options'] = {'include_usage': True}
                with client.chat.completions.stream(**stream_args) as stream:
                    for event in stream:
                        if event.type == 'chunk' and 'first_meaningful_output_seconds' not in row:
                            if any(choice.delta.content or choice.delta.tool_calls
                                   for choice in event.chunk.choices):
                                row['first_meaningful_output_seconds'] = time.monotonic() - started
                    completion = stream.get_final_completion()
            else:
                completion = client.chat.completions.create(**body)
            result = completion.model_dump(mode='json', exclude_none=True)
            save(args.output / (name + '.response.json'), result)
            choice, = result['choices']
            message = choice['message']
            if result['model'] != MODEL or not result['id']:
                raise ValueError('Response identity differs from the requested model')
            if choice['finish_reason'] not in (('tool_calls', 'stop') if tool else ('stop',)):
                raise ValueError('Response did not complete naturally')
            usage = result['usage']
            if not (0 < usage['completion_tokens'] < body['max_tokens'] and usage['prompt_tokens'] > 0
                    and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']):
                raise ValueError('Missing or inconsistent completed-response usage')
            if expected is not None and (message.get('content') or '').strip() != expected:
                raise ValueError('Final answer differs from the oracle')
            if json_expected is not None and json.loads(message['content']) != json_expected:
                raise ValueError('Structured answer differs from the oracle')
            if tool:
                call, = message['tool_calls']
                if (not call['id'] or call['function']['name'] != TOOL_NAME
                        or json.loads(call['function']['arguments']) != TOOL_ARGUMENTS):
                    raise ValueError('Incorrect tool call')
            row.update(passed=True, usage=usage, finish_reason=choice['finish_reason'])
            return result
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            row['completed_response_seconds'] = time.monotonic() - started
            save(args.output / 'summary.json', report)

    try:
        if MODEL not in [model.id for model in client.models.list().data]:
            raise ValueError('The intended model is absent from /v1/models')
        for effort in REASONING_EFFORTS:
            for streaming in (False, True):
                body = payload('Calculate (37 * 19) + 11. Reply only with the decimal integer.',
                               stream=streaming, effort=effort)
                request(f'reasoning-{effort}-{streaming}', body, expected='714')
        expected = {'status': 'ok', 'count': 3}
        for streaming in (False, True):
            body = payload('Return exactly {"status":"ok","count":3}.', stream=streaming)
            body['response_format'] = schema(expected)
            request(f'json-{streaming}', body, json_expected=expected)
        body = payload('Use lookup_inventory to look up SKU INK2-LAB. Do not guess. After the '
                       'tool result, reply exactly Inventory: followed by one space and the '
                       'available count.', stream=True)
        body.update(tools=TOOLS, tool_choice='auto', parallel_tool_calls=False)
        result = request('auto-tool-stream', body, tool=True)
        assistant = deepcopy(result['choices'][0]['message'])
        followup = deepcopy(body)
        followup.update(stream=False, tool_choice='none')
        followup['messages'] += [assistant, {'role': 'tool',
                                             'tool_call_id': assistant['tool_calls'][0]['id'],
                                             'content': json.dumps(TOOL_RESULT)}]
        request('complete-assistant-tool-roundtrip', followup, expected=TOOL_ANSWER)
        manifest = json.loads((args.fixtures / 'manifest.json').read_text())
        for image in manifest['cases']:
            raw = (args.fixtures / image['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != image['sha256']:
                raise ValueError('Image fixture hash differs from its manifest')
            body = payload([{'type': 'text', 'text': 'Identify the color in each quadrant. '
                             'Return JSON with top_left, top_right, bottom_left and '
                             'bottom_right, each a color name.'},
                            {'type': 'image_url', 'image_url':
                             {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}}])
            body['response_format'] = schema(image['expected'])
            request('image-' + image['file'], body, json_expected=image['expected'])
        prefix = '\n'.join(f'Reference row {i}: the lab marker is INK2_PREFIX_OK.'
                           for i in range(PREFIX_ROWS))
        for suffix in ('First question', 'Second question'):
            result = request('long-prefix-' + suffix.split()[0],
                             payload(prefix + '\n' + suffix + ': return only the lab marker.'),
                             expected='INK2_PREFIX_OK')
        cached = result['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if cached <= 0:
            raise ValueError('The repeated long prefix did not exercise cached tokens')
        body = payload('Write a long Python tutorial on iterators and generators.', stream=True)
        body['max_tokens'] = CANCEL_TOKEN_LIMIT
        emitted = 0
        with client.chat.completions.create(**body) as stream:
            for chunk in stream:
                emitted += int(any(choice.delta.content
                                   or getattr(choice.delta, 'reasoning_content', None)
                                   or getattr(choice.delta, 'reasoning', None)
                                   for choice in chunk.choices))
                if emitted >= CANCEL_EVENT_COUNT:
                    break
        if emitted < CANCEL_EVENT_COUNT:
            raise ValueError('The cancellation request ended before explicit closure')
        report['cases'].append({'name': 'stream-cancel', 'passed': True,
                                'events_before_close': emitted})
        request('reuse-after-cancel', payload('Reply exactly INK2_REUSE_OK.'),
                expected='INK2_REUSE_OK')
        report['passed'] = True
    finally:
        client.close()
        save(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
