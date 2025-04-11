from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from abc import abstractmethod
from typing import Any

import numpy as np
import math
import json

'''
    Logger
'''
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

'''
    Strategies    
'''
class Strategy:
    def __init__(self, symbol: str, max_position: int):
        self.symbol = symbol
        self.max_position = max_position
        self.true_value = 0.0
    
    def trade(self, state: TradingState) -> list[Order]:
        self.orders: list[Order] = []
        self.position = state.position.get(self.symbol, 0)
        self.aggregate_buy_quantity = 0
        self.aggregate_sell_quantity = 0

        self.true_value = self.get_true_value(state)
        self.run(state)

        return self.orders

    @abstractmethod
    def run(self, state: TradingState):
        raise NotImplementedError()

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        return NotImplementedError()

    def buy_product(self, price: int, target_quantity: int) -> None:
        """
        Executes a BUY order for this product.
        Attempts to buy for the entire target quantity (without exceeding self.max_position).

        :param price: Price to use for the order
        :param target_quantity: Target quantity of the order (should be > 0)
        """
        if target_quantity <= 0:
            return
        
        quantity = min(self.position + target_quantity, self.max_position - self.aggregate_buy_quantity) - self.position
        if quantity == 0:
            return

        self.orders.append(Order(self.symbol, price, quantity))
        self.aggregate_buy_quantity += quantity

    def sell_product(self, price: int, target_quantity: int) -> None:
        """
        Executes a SELL order for this product.
        Attempts to sell for the entire target quantity (without exceeding self.max_position).

        :param price: Price to use for the order
        :param target_quantity: Target quantity of the order (should be < 0)
        """ 
        if target_quantity >= 0:
            return
        
        quantity = max(self.position + target_quantity, -self.max_position - self.aggregate_sell_quantity) - self.position
        if quantity == 0:
            return
        
        self.orders.append(Order(self.symbol, price, quantity))
        self.aggregate_sell_quantity += quantity
    
    def force_buy(self, order_depth: OrderDepth, target_quantity: int) -> None:
        if target_quantity <= 0 or len(order_depth.sell_orders) == 0:
            return
        
        for ask, quantity in sorted(order_depth.sell_orders.items(), key=lambda item: item[0]):
            if target_quantity > 0:
                buy_quantity = min(target_quantity, -quantity)
                self.buy_product(ask, buy_quantity)
                target_quantity -= buy_quantity

    def force_sell(self, order_depth: OrderDepth, target_quantity: int) -> None:
        if target_quantity <= 0 or len(order_depth.buy_orders) == 0:
            return
                
        for bid, quantity in sorted(order_depth.buy_orders.items(), key=lambda item: -item[0]):
            if target_quantity > 0:
                sell_quantity = min(target_quantity, quantity)
                self.sell_product(bid, -sell_quantity)
                target_quantity -= sell_quantity

class MM_Fixed(Strategy):
    def __init__(self, symbol: str, max_position: int, take_width: int, make_width: int, make_volume: int):
        super().__init__(symbol, max_position)
        self.take_width = take_width
        self.make_width = make_width
        self.make_volume = make_volume
    
    def run(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        buy_orders, sell_orders = order_depth.buy_orders, order_depth.sell_orders

        self.market_take(buy_orders, sell_orders)
        self.market_make()
    
    def market_take(self, buy_orders: dict[int, int], sell_orders: dict[int, int]) -> None:
        true_value = round(self.true_value)

        # Take good ask prices
        if len(sell_orders) > 0:
            for ask, quantity in sorted(sell_orders.items(), key=lambda item: item[0]):
                if ask <= true_value - self.take_width:
                    self.buy_product(ask, -quantity)
        
        # Take good bid prices
        if len(buy_orders) > 0:
            for bid, quantity in sorted(buy_orders.items(), key=lambda item: -item[0]):
                if bid >= true_value + self.take_width:
                    self.sell_product(bid, -quantity)

    def market_make(self) -> None:
        true_value = round(self.true_value)
        self.sell_product(true_value + self.make_width, -self.make_volume)
        self.buy_product(true_value - self.make_width, self.make_volume)

class MM_GLFT(Strategy):
    '''
    Guéant Lehalle Fernandez-Tapia Market Making Model
    https://hftbacktest.readthedocs.io/en/py-v2.0.0/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html
    '''
    def __init__(self, symbol: str, max_position: int, trade_volume: int, sigma: int, gamma: int, kappa: int, A: int, xi: int):
        """
        :param sigma: Price volatility
        :param gamma: Market impact parameter
        :param kappa: Market order arrival intensity
        :param A: Liquidity parameter
        :param xi: Inventory risk aversion
        """ 
        super().__init__(symbol, max_position)
        self.trade_volume = trade_volume
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = kappa
        self.A = A
        self.xi = xi
    
    def run(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        
        q = self.position / self.max_position

        sigma, gamma, kappa, A, xi = self.sigma, self.gamma, self.kappa, self.A, self.xi

        c1 = 1 / xi * math.log(1 + xi / kappa)
        c2 = sigma * math.sqrt(gamma / (2 * A * kappa) * (1 + xi / kappa)**(kappa / xi + 1))

        delta_b = c1 + 1 / 2 * c2 + q * c2
        delta_a = c1 + 1 / 2 * c2 - q * c2

        optimal_bid = round(self.true_value - delta_b)
        optimal_ask = round(self.true_value + delta_a)

        # Market take good ask prices
        if len(order_depth.sell_orders) != 0:
            for ask, quantity in sorted(order_depth.sell_orders.items(), key=lambda item: item[0]):
                if ask < optimal_ask:
                    self.buy_product(ask, -quantity)
                    
        # Market take good bid prices
        if len(order_depth.buy_orders) != 0:
            for bid, quantity in sorted(order_depth.buy_orders.items(), key=lambda item: -item[0]):
                if bid > optimal_bid:
                    self.sell_product(bid, -quantity)

        # Market make
        self.buy_product(optimal_bid, self.trade_volume)
        self.sell_product(optimal_ask, -self.trade_volume)

class Mean_Reversion(Strategy):
    def __init__(self, symbol: str, max_position: int, window_size: int, z_score_hard: float, z_score_soft: float):
        super().__init__(symbol, max_position)
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.z_score_hard = z_score_hard
        self.z_score_soft = z_score_soft
        self.prev_ema = None

    def run(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        true_value = round(self.true_value)

        self.window.append(self.true_value)
        if len(self.window) < self.window_size:
            logger.print(f"WAITING FOR MORE DATA: {len(self.window)}/{self.window_size}")
            return

        ema = self.calculate_ema(self.window_size)
        z_score = self.calculate_z_score(ema)

        # Mean reversion signals
        if z_score > self.z_score_hard:
            max_sell_quantity = self.max_position + self.position
            self.force_sell(order_depth, max_sell_quantity)
        elif z_score > self.z_score_soft:
            self.sell_product(true_value, -6)
        elif z_score < -self.z_score_hard:
            max_buy_quantity = self.max_position - self.position
            self.force_buy(order_depth, max_buy_quantity)
        elif z_score < -self.z_score_soft:
            self.buy_product(true_value, 6)

        # Exit strategy
        if self.position / self.max_position >= 0.75 and z_score > -0.6:
            self.sell_product(true_value, -4)
        elif self.position / self.max_position <= -0.75 and z_score < 0.6:
            self.buy_product(true_value, 4)
        elif self.position > 0 and z_score > 0:
            self.sell_product(true_value, -6)
        elif self.position < 0 and z_score < 0:
            self.buy_product(true_value, 6)

        logger.print("z_score:", z_score)
        logger.print("ema:", ema)
        logger.print("true_value:", true_value)

        self.prev_ema = ema
    
    def calculate_ema(self, period: int) -> float:
        multipler = 2 / (period + 1)
        if self.prev_ema == None:
            return sum(self.window) / period
        else:
            return self.window[-1] * multipler + self.prev_ema * (1 - multipler)
    
    def calculate_z_score(self, mean: float) -> float:
        std = np.std(self.window)
        if std == 0:
            return 0
        return (self.true_value - mean) / std
    

'''
    Products
'''
class RainforestResin(MM_Fixed):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", max_position=50, take_width=1, make_width=4, make_volume=12)
    
    def get_true_value(self, state: TradingState) -> float:
        return 10_000

class Kelp(MM_GLFT):
    def __init__(self):
        super().__init__("KELP", max_position=50, trade_volume=16, sigma=0.4, gamma=0.65, kappa=2, A=0.1, xi=1)
    
    def get_true_value(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        vwa_ask = np.average(list(order_depth.sell_orders.keys()), weights=list(order_depth.sell_orders.values()))
        vwa_bid = np.average(list(order_depth.buy_orders.keys()), weights=list(order_depth.buy_orders.values()))
        return (vwa_ask + vwa_bid) / 2

class SquidInk(Mean_Reversion):
    def __init__(self):
        super().__init__("SQUID_INK", max_position=50, window_size=100, z_score_hard=4, z_score_soft=2)
    
    def get_true_value(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        vwa_ask = np.average(list(order_depth.sell_orders.keys()), weights=list(order_depth.sell_orders.values()))
        vwa_bid = np.average(list(order_depth.buy_orders.keys()), weights=list(order_depth.buy_orders.values()))
        return (vwa_ask + vwa_bid) / 2

'''
    Trader Logic
'''
class Trader:
    def __init__(self):
        self.products: dict[Symbol, Strategy] = {
            "RAINFOREST_RESIN": RainforestResin(),
            "KELP": Kelp(),
            "SQUID_INK": SquidInk(),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        trader_data = ""

        for symbol, strategy in self.products.items():
            if symbol in state.order_depths.keys():
                result[symbol] = strategy.trade(state)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
