from asr2rpp.text_join import join_timed


def test_no_spaces_inside_subwords():
    assert join_timed('hel', 'lo', 'token') == 'hello'
    assert join_timed('hello', '\u2581world', 'token') == 'hello world'


def test_lexical_words_and_japanese():
    assert join_timed('hello', 'world', 'word') == 'hello world'
    assert join_timed('こんにちは', '世界', 'word') == 'こんにちは世界'
