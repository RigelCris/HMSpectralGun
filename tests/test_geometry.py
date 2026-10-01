import tempfile
import unittest
from pathlib import Path

from InputParameters import InputParameters
from ModelMaker import ModelMaker, SkipModelError
from TurboSpecWriter import TurboSpecWriter
from model_geometry import atmosphere_geometry


class GeometryTests(unittest.TestCase):
    def test_input_and_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.ts'
            base = '/tmp/out\n/tmp/lines\n/tmp/models\nExplicitModel=False\ninterp=True\nNLTE=False\n'
            row = '5875,4.15 -0.3 0.3 8000 13000 1.5 st * 2700 0.02 * lines abu * txt\n'
            for setting, expected in [('', 'auto'), ('geometry=plane-parallel\n', 'plane-parallel'), ('geometry=spherical\n', 'spherical')]:
                path.write_text(base + setting + row)
                params = InputParameters(path)
                self.assertEqual(params.geometry, expected)
                self.assertEqual(len(params.df), 1)
            path.write_text(base + 'geometry=typo\n' + row)
            with self.assertRaises(ValueError):
                InputParameters(path)

    def test_explicit_alias_and_default_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.ts'
            row = '5875,4.15 -0.3 0.3 8000 13000 1.5 st * 2700 0.02 * lines abu * txt\n'
            for keyword in ('ExplicitModel', 'Explicit', 'explicit'):
                path.write_text(
                    f'/tmp/out\n/tmp/lines\n/tmp/models\n{keyword}=False\n'
                    f'interp=True\nNLTE=False\n{row}'
                )
                params = InputParameters(path)
                self.assertEqual(params.get_keywords(), ('False', 'True', 'False'))
                self.assertEqual(params.geometry, 'auto')
                self.assertEqual(len(params.df), 1)

    def test_interpolation_keyword_normalization_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.ts'
            for value, expected in [('interp', 'True'), ('true', 'True'), (' TRUE ', 'True'), ('nearest', 'Nearest')]:
                path.write_text(f'/tmp/out\n/tmp/lines\n/tmp/models\nExplicitModel=false\ninterp={value}\nNLTE=false\n')
                params = InputParameters(path)
                self.assertEqual(params.get_keywords(), ('False', expected, 'False'))
                self.assertEqual(params.geometry, 'auto')
            for explicit, interp in [('False', 'typo'), ('False', 'False'), ('True', 'True')]:
                path.write_text(f'/tmp/out\n/tmp/lines\n/tmp/models\nExplicitModel={explicit}\ninterp={interp}\nNLTE=False\n')
                with self.assertRaises(ValueError):
                    InputParameters(path)

    def test_selection_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / 'grid'
            dataset.mkdir()
            for prefix, mass in [('p', '0.0'), ('s', '1.0')]:
                for teff in [5750, 6000]:
                    for gravity in [4.0, 4.5]:
                        for met in [-0.5, 0.0]:
                            name = f'{prefix}{teff}_g+{gravity:.1f}_m{mass}_t02_st_z{met:+.2f}.mod'
                            (dataset / name).write_text(name + '\n')
            for geometry, prefix in [('plane-parallel', 'p'), ('spherical', 's'), ('auto', 'p')]:
                maker = ModelMaker(str(dataset), geometry=geometry)
                models = maker.select_models_for_interpolation(5875, 4.15, -0.3, 1.5, 'st')
                self.assertEqual(len(models), 8)
                self.assertTrue(all(m.startswith(prefix) for m in models))
                nearest = maker.select_nearest_model(5875, 4.15, -0.3, 1.5, 'st', Path(temp) / 'out')
                self.assertEqual(atmosphere_geometry(Path(temp) / 'out' / nearest), 'plane-parallel' if prefix == 'p' else 'spherical')
                mixed = list(models)
                mixed[0] = next(dataset.glob(('s' if prefix == 'p' else 'p') + '*.mod')).name
                with self.assertRaises(ValueError):
                    maker.write_interpolator(5875, 4.15, -0.3, 1.5, 'st', Path(temp) / 'out', mixed)
            for path in dataset.glob('p*.mod'):
                path.unlink()
            with self.assertRaises(SkipModelError):
                ModelMaker(str(dataset), geometry='plane-parallel').select_models_for_interpolation(5875, 4.15, -0.3, 1.5, 'st')

    def test_writer_uses_actual_geometry_even_across_gravity_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = TurboSpecWriter(temp, temp, temp, temp)
            for geometry, tag, gravity, header, flag in [
                ('plane-parallel', 'p', '3.50', "'ppINTERPOL' 56", 'F'),
                ('spherical', 's', '4.15', "'sphINTERPOL' 56", 'T'),
            ]:
                model = f'T5875_Gp{gravity}_{tag}_st_Zm0.30_xi1.50.interpol'
                (root / model).write_text(header + '\n')
                name = writer.writer(model, -0.3, 0.3, 8000, 8010, 1.5, [], [], [0], [], 612613, 89, 'No', '*', 'txt', geometry=geometry)
                self.assertEqual(
                    name,
                    f'T5875_Gp{gravity}_8000_8010_xi1.50_zm0.30_ap0.30.txt',
                )
                self.assertIn(f"'SPHERICAL:'  '{flag}'", (root / (name + '.com')).read_text())
                with self.assertRaises(ValueError):
                    atmosphere_geometry(root / model, 'spherical' if tag == 'p' else 'plane-parallel')

    def test_nearest_writer_keeps_historical_spectrum_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = TurboSpecWriter(temp, temp, temp, temp)
            model = 'T5875_Gp4.15_p5750_g+4.0_m0.0_t02_st_z-0.50.mod'
            (root / model).write_text('p5750 test atmosphere\n')
            name = writer.writer(
                model, -0.3, 0.3, 8000, 8010, 1.5, [], [], [0], [],
                612613, 89, 'No', '*', 'txt', interp='Nearest', geometry='auto'
            )
            self.assertEqual(
                name,
                'T5875_Gp4.15_8000_8010_xi1.50_zm0.30_ap0.30.txt',
            )


if __name__ == '__main__':
    unittest.main()
