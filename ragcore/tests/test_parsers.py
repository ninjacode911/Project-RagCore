from app.utils.parsers import parse_document, parse_text, parse_html


def test_parse_text_utf8():
    content = "Hello, world! This is a test."
    result = parse_text(content.encode("utf-8"), "test.txt")
    assert "Hello, world" in result


def test_parse_text_latin1():
    content = "Héllo wörld"
    result = parse_text(content.encode("latin-1"), "test.txt")
    assert "rld" in result


def test_parse_html():
    html = b"<html><body><p>Hello world</p><script>var x=1;</script></body></html>"
    result = parse_html(html, "test.html")
    assert "Hello world" in result
    assert "var x" not in result


def test_parse_document_unsupported():
    result = parse_document(b"data", "test.xyz")
    assert result == ""


def test_parse_empty_text():
    result = parse_text(b"", "empty.txt")
    assert result == ""


def test_parse_document_dispatches_by_extension():
    result = parse_document(b"Hello text file", "readme.txt")
    assert "Hello text file" in result
