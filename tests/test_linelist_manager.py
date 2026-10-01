import tempfile
import unittest
from pathlib import Path

from LineListManager import LineListManager


class SingleElementLineListTests(unittest.TestCase):
    def test_optical_cn_isotopologue_names_are_selected(self):
        lines = [
            'OPT_molec/C12N14.list',
            'OPT_molec/C13N15.list',
            'OPT_molec/12CH.list',
            'OPT_molec/OH_HITRAN-IR.list',
        ]
        self.assertEqual(
            LineListManager.use_only_molecular_data('', lines, ['*', '*', 'CN']),
            ['OPT_molec/C12N14.list', 'OPT_molec/C13N15.list'],
        )

    def test_padded_single_digit_atomic_number_and_absent_segment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            present = root / "vald-present.list"
            absent = root / "vald-absent.list"
            present.write_text(
                "'   8.000            '    1         2\n"
                "'O I   '\n"
                "  6300.000  0.000 -1.000\n"
                "  6363.000  0.000 -1.000\n"
            )
            absent.write_text(
                "'  26.000            '    1         1\n"
                "'Fe I  '\n"
                "  6301.000  0.000 -1.000\n"
            )

            result = LineListManager.create_linelist_single_element(
                str(root), [present.name, absent.name], ['8.000', '1', 'OI'],
                tmp_tag='test',
            )

            self.assertEqual(result, ['tmp_test_vald-present.OI'])
            content = (root / result[0]).read_text()
            self.assertIn("'   8.000", content)
            self.assertIn('6300.000', content)
            self.assertNotIn('Fe I', content)

            LineListManager.delete_tmp_linelist(
                str(root), result, ['8.000', '1', 'OI'], keyw='Yes'
            )
            self.assertFalse((root / result[0]).exists())


if __name__ == '__main__':
    unittest.main()
