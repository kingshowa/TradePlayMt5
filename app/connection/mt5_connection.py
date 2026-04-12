import MetaTrader5 as mt5
from typing import Optional


class MT5Connection:
    """
    Global MT5 connection manager.
    Ensures single active connection across the entire system.
    """

    _initialized: bool = False
    _login: Optional[int] = None
    _server: Optional[str] = None

    # ----------------------------------
    # Initialization
    # ----------------------------------

    @classmethod
    def initialize(
        cls,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None
    ) -> None:
        """
        Initialize MT5 connection.
        Safe to call multiple times.
        """

        if cls._initialized:
            return

        if path:
            success = mt5.initialize(path=path)
        else:
            success = mt5.initialize()

        if not success:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        # Optional login
        if login:
            authorized = mt5.login(login, password=password, server=server)
            if not authorized:
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

            cls._login = login
            cls._server = server

        cls._initialized = True

    # ----------------------------------
    # Status Check
    # ----------------------------------

    @classmethod
    def ensure_connection(cls) -> None:
        """
        Ensure connection is alive.
        Reconnect automatically if needed.
        """

        if not cls._initialized:
            raise RuntimeError("MT5 not initialized. Call initialize() first.")

        if not mt5.terminal_info():
            # Connection lost → reinitialize
            mt5.shutdown()
            cls._initialized = False
            cls.initialize(cls._login, server=cls._server)

    # ----------------------------------
    # Symbol Helper
    # ----------------------------------

    @staticmethod
    def select_symbol(symbol: str) -> None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}")

    # ----------------------------------
    # Shutdown
    # ----------------------------------

    @classmethod
    def shutdown(cls) -> None:
        if cls._initialized:
            mt5.shutdown()
            cls._initialized = False