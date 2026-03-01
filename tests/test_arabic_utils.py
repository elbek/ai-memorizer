import pytest
from benchmark.arabic_utils import strip_tashkeel, normalize_arabic


class TestStripTashkeel:
    """Tests for strip_tashkeel function."""

    def test_empty_string(self):
        assert strip_tashkeel("") == ""

    def test_no_diacritics_passthrough(self):
        text = "بسم الله الرحمن الرحيم"
        assert strip_tashkeel(text) == text

    def test_fathah(self):
        assert strip_tashkeel("بَ") == "ب"

    def test_dammah(self):
        assert strip_tashkeel("بُ") == "ب"

    def test_kasrah(self):
        assert strip_tashkeel("بِ") == "ب"

    def test_sukun(self):
        assert strip_tashkeel("بْ") == "ب"

    def test_shadda(self):
        assert strip_tashkeel("بّ") == "ب"

    def test_tanwin_fath(self):
        assert strip_tashkeel("بً") == "ب"

    def test_tanwin_damm(self):
        assert strip_tashkeel("بٌ") == "ب"

    def test_tanwin_kasr(self):
        assert strip_tashkeel("بٍ") == "ب"

    def test_superscript_alef(self):
        # \u0670 superscript alef
        assert strip_tashkeel("بٰ") == "ب"

    def test_full_basmala(self):
        basmala = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        expected = "بسم الله الرحمن الرحيم"
        assert strip_tashkeel(basmala) == expected

    def test_combined_diacritics(self):
        # shadda + fathah
        assert strip_tashkeel("لَّ") == "ل"

    def test_latin_text_unchanged(self):
        assert strip_tashkeel("hello world") == "hello world"

    def test_mixed_arabic_latin(self):
        assert strip_tashkeel("بِسْمِ hello") == "بسم hello"

    def test_presentation_forms(self):
        # \uFE76 ARABIC FATHATAN ISOLATED FORM
        text = "ب\uFE76"
        assert strip_tashkeel(text) == "ب"

    def test_extended_arabic_marks_0610_061A(self):
        # \u0610 ARABIC SIGN SALLALLAHOU ALAYHE WASSALLAM
        text = "ب\u0610"
        assert strip_tashkeel(text) == "ب"

    def test_quranic_annotation_marks(self):
        # \u06D6 ARABIC SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA
        text = "ب\u06D6"
        assert strip_tashkeel(text) == "ب"

    def test_marks_06DF_06E4(self):
        text = "ب\u06E1"
        assert strip_tashkeel(text) == "ب"

    def test_marks_06E7_06E8(self):
        text = "ب\u06E7"
        assert strip_tashkeel(text) == "ب"

    def test_marks_06EA_06ED(self):
        text = "ب\u06EB"
        assert strip_tashkeel(text) == "ب"


class TestNormalizeArabic:
    """Tests for normalize_arabic function."""

    def test_empty_string(self):
        assert normalize_arabic("") == ""

    def test_strips_tashkeel(self):
        assert normalize_arabic("بِسْمِ") == "بسم"

    def test_removes_tatweel(self):
        # \u0640 ARABIC TATWEEL
        assert normalize_arabic("كتـــاب") == "كتاب"

    def test_normalizes_alef_madda(self):
        # \u0622 ARABIC LETTER ALEF WITH MADDA ABOVE -> \u0627
        assert normalize_arabic("آ") == "ا"

    def test_normalizes_alef_hamza_above(self):
        # \u0623 ARABIC LETTER ALEF WITH HAMZA ABOVE -> \u0627
        assert normalize_arabic("أ") == "ا"

    def test_normalizes_alef_hamza_below(self):
        # \u0625 ARABIC LETTER ALEF WITH HAMZA BELOW -> \u0627
        assert normalize_arabic("إ") == "ا"

    def test_normalizes_alef_wasla(self):
        # \u0671 ARABIC LETTER ALEF WASLA -> \u0627
        assert normalize_arabic("ٱ") == "ا"

    def test_teh_marbuta_to_heh(self):
        # \u0629 -> \u0647
        assert normalize_arabic("رحمة") == "رحمه"

    def test_removes_arabic_comma(self):
        assert normalize_arabic("بسم، الله") == "بسم الله"

    def test_removes_arabic_question_mark(self):
        assert normalize_arabic("ما هذا؟") == "ما هذا"

    def test_removes_arabic_semicolon(self):
        assert normalize_arabic("بسم؛ الله") == "بسم الله"

    def test_removes_period(self):
        assert normalize_arabic("بسم. الله") == "بسم الله"

    def test_removes_exclamation(self):
        assert normalize_arabic("بسم! الله") == "بسم الله"

    def test_removes_colon(self):
        assert normalize_arabic("بسم: الله") == "بسم الله"

    def test_collapses_whitespace(self):
        assert normalize_arabic("بسم   الله") == "بسم الله"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_arabic("  بسم الله  ") == "بسم الله"

    def test_full_normalization(self):
        """Test all normalization steps combined."""
        text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ."
        expected = "بسم الله الرحمن الرحيم"
        assert normalize_arabic(text) == expected

    def test_no_change_needed(self):
        text = "بسم الله"
        assert normalize_arabic(text) == text

    def test_complex_normalization(self):
        """Alef variants + teh marbuta + tatweel + diacritics + punctuation."""
        text = "أُمَّةٌ، إِسْلَامِيَّةٌ."
        expected = "امه اسلاميه"
        assert normalize_arabic(text) == expected
