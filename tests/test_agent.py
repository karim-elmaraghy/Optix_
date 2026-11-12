import json
import os
import tempfile
import unittest
from unittest.mock import patch
from tunix.agent import meta_completion, optimize, parse_proposal, propose

C = dict(dtype='float16', attn='sdpa', batch=1, compile=False, max_new_tokens=128)


class AgentTests(unittest.TestCase):
    def test_parse(self):
        text = json.dumps(dict(config=C, reason='Measured candidate'))
        self.assertEqual(parse_proposal(text, [C])['config'], C)
        for bad in ('no json', '{}', json.dumps(dict(config=C, reason='')),
                    json.dumps(dict(config=dict(C, batch=True), reason='x'))):
            with self.assertRaises((ValueError, TypeError, KeyError)):
                parse_proposal(bad, [C])
        with self.assertRaises(ValueError):
            parse_proposal(text, [])

    def test_provenance(self):
        good = lambda messages: json.dumps(dict(config=C, reason='Explore'))
        self.assertEqual(propose([C], [], good)['source'], 'meta_api')
        self.assertEqual(propose([C], [], lambda _: '{}')['source'], 'rule_fallback')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(propose([C], [])['source'], 'rule_fallback')

    def test_http_transport(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': 'reply'}}]}).encode()
        with patch.dict(os.environ, {'MODEL_API_KEY': 'fixture-key'}), patch('urllib.request.urlopen', return_value=Response()) as call:
            self.assertEqual(meta_completion([]), 'reply')
            request = call.call_args.args[0]
            self.assertEqual(request.full_url, 'https://api.meta.ai/v1/chat/completions')
            self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-key')

    def test_rejections_logged_and_budget_respected(self):
        space = [C, dict(C, batch=4)]
        def evaluate(config):
            return dict(benchmark=dict(status='success', requested_config=config, actual_config=config),
                        validation=dict(accepted=False, reasons=['truncation']))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'events.jsonl')
            history = optimize(space, evaluate, budget=8, completion=lambda _: '{}', log_path=path)
            self.assertEqual(len(history), 2)
            self.assertTrue(all(not r['accepted'] for r in history))
            with open(path) as stream:
                self.assertEqual(json.loads(stream.readline())['reasons'], ['truncation'])

    def test_execution_exception(self):
        def fail(_): raise RuntimeError('secret detail')
        result = optimize([C], fail, completion=lambda _: '{}')[0]
        self.assertEqual(result['reasons'], ['evaluation_error'])
        self.assertNotIn('secret detail', json.dumps(result))
