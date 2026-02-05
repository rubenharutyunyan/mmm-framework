from mmm.config.schema import infer_role, parse_column_name


def test_infer_role():
    assert infer_role("media__tv__spend") == "media"
    assert infer_role("target__sales") == "target"
    assert infer_role("date") == "date"
    assert infer_role("TV Spend") is None


def test_parse_column_name():
    p = parse_column_name("media__tv__spend")
    assert p is not None
    assert p.role == "media"
    assert p.parts == ("tv", "spend")
