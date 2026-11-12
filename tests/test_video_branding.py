"""Presentation-only contract checks; existing backend gates stay untouched."""
import unittest
import app
from tunix import workbench_ui as ui


class VideoBrandingTests(unittest.TestCase):
    def test_install_reveal_and_copyable_real_commands(self):
        demo=app.create_app(); config=demo.get_config_file()
        button=next(c for c in config['components'] if c['type']=='button' and c['props'].get('value')=='Install Tunix')
        dependency=next(d for d in config['dependencies'] if any(t[0]==button['id'] and t[1]=='click' for t in d['targets']))
        self.assertTrue(demo.fns[dependency['id']].fn()['visible'])
        code=next(c for c in config['components'] if c['type']=='code' and c['props'].get('label')=='Copy installation commands')
        self.assertIn('copy',code['props']['buttons'])
        self.assertIn('git clone https://github.com/karim-elmaraghy/Tunix.git',code['props']['value'])
        self.assertIn('python -m pip install -e .',code['props']['value'])
        self.assertNotIn('pip install tunix',code['props']['value'])
        upload=next(c for c in config['components'] if c['type']=='file' and 'Import ZIP / Browse files' in c['props'].get('label',''))
        self.assertEqual(upload['props']['file_types'],['.zip'])

    def test_brand_system_and_landing_approval_separation(self):
        self.assertIn('#A36545',ui.MARK)
        self.assertIn('M44 18 H66',ui.MARK)
        self.assertNotIn('@import',ui.CSS)
        self.assertNotIn('fonts.googleapis',ui.CSS)
        cards,_=ui.stats_and_charts(app.load_dashboard_data(follow_session=False),landing=True)
        self.assertIn('5.63',cards)
        self.assertIn('81.8%',cards)
        self.assertNotIn('Blocked',cards)
        self.assertNotIn('QUALITY APPROVAL',cards)
        panel=ui.status_panel(app.load_dashboard_data(follow_session=False))
        self.assertIn('Approval withheld',panel)
        self.assertIn('5 of 20',panel)
        self.assertIn('calibration remains BLOCKED',panel)
