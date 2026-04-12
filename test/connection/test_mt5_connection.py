import MetaTrader5 as mt5
import pytest
from app.connection.mt5_connection import MT5Connection

class TestMT5Connection:

    @classmethod
    def setup_class(cls):
        MT5Connection.initialize()

    @classmethod
    def teardown_class(cls):
        MT5Connection.shutdown()

    def test_initialize(self):
        assert mt5.terminal_info() is not None

    def test_symbol_selection(self):
        MT5Connection.select_symbol("XAUUSDm")
        info = mt5.symbol_info("XAUUSDm")
        assert info is not None
        assert info.visible is True

    def test_ensure_connection(self):
        MT5Connection.ensure_connection()
        assert mt5.terminal_info() is not None