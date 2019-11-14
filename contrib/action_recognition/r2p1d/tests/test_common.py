from vu.utils.common import Config, system_info


def test_config():
    cfg = Config({'a': 1}, b=2, c=3)
    assert cfg.a == 1 and cfg.b == 2 and cfg.c == 3

    cfg = Config({'a': 1, 'b': 2})
    assert cfg.a == 1 and cfg.b == 2

    cfg2 = Config(cfg)
    assert cfg2.a == cfg.a and cfg2.b == cfg.b


def test_system_info():
    system_info()
