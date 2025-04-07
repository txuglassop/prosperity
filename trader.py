from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, TypeAlias
import string
from collections import deque
from abc import ABC, abstractmethod
import json

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

##################################################################################################################################
##################################################################################################################################
################################################## LOGGER - NOT FOR ACTUAL TRADING ###############################################
##################################################################################################################################
##################################################################################################################################


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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
                observation.sunlight,
                observation.humidity,
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

        return value[:max_length - 3] + "..."

logger = Logger()

##################################################################################################################################
##################################################################################################################################
################################################### TRADING STRATEGIES ###########################################################
##################################################################################################################################
##################################################################################################################################


class Strategy:
    """
    A generic trading class

    need to define the act method for any instance of this class, which
    defines how it takes in TradingState and makes trades

    In any instance of this class, use the `self.buy` and `self.sell` to send
    buy/sell orders when creating the `run` method.
    """
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class MarketMakingStrategy(Strategy):
    """
    A market making strategy. Sends buy/sell orders around a theo, given by the
    `get_true_value` method
    """
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        buy_spread = 1,  # premium charged on theo when we are in an excessively long/short position
        sell_spread = 1,
        buy_tolerance = 0.5, # our tolerance as to what constitutes an "excessively long/short position"
        sell_tolerance = 0.5
    ) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 100 # sensitivity to limit positions
        self.buy_spread = buy_spread
        self.sell_spread = sell_spread
        self.buy_tolerance = buy_tolerance
        self.sell_tolerance = sell_tolerance

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()
    
    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # find out how many items we can buy/sell
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # keep track of the last 10 states, recording True if our position
        # was at the position limit
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        ### define liquidity actions
        # if we've observed 10 periods AND 5 of these times we've been at our limit AND the most recent period was at the limit
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]

        # if we've observed 10 periods AND all 10 of those we were at a limit
        hard_liquidate = len(self.window) == self.window_size and all(self.window)
        ###

        # if we already have a few of the asset, have a slightly lower max buy price
        max_buy_price = true_value - self.buy_spread if position > self.limit * self.buy_tolerance else true_value
        # if we are already short a few, have a slightly higher min sell price
        min_sell_price = true_value + self.sell_spread if position < self.limit * -self.sell_tolerance else true_value

        # go through all the price/volume in sell_orders
        for price, volume in sell_orders:
            # if we are in a position to buy and the price is right, buy
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if we are in a position to buy and we need to liquidate BADLY!
        if to_buy > 0 and hard_liquidate:
            # put out a bunch of buy orders
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        # if we are in a position to buy and we need to liquidate KINDA BAD
        if to_buy > 0 and soft_liquidate:
            # put out a bunch of buy orders but we're not down bad on price
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        # if we are in a position to buy
        if to_buy > 0:
            # the "popular price" is the price corresponding to the order w/ the least orders
            # if the popular buy price is below our theo, go long
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)
        
        # go through all the price, volume in buy orders 
        for price, volume in buy_orders:
            # if we are in a position to sell and we can sell above our min
            # sell to all of these bids
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        # if we are in a position to sell where we need to liquidate BADLY!
        if to_sell > 0 and hard_liquidate:
            # put out a bunch of sell orders up to half our potential sells
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        # if we are in a position to sell where we need to liquidate KINDA BAD
        if to_sell > 0 and soft_liquidate:
            # put out a bunch of sell orders but not down as bad on price
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        # if we are in a position to sell
        if to_sell > 0:
            # the "popular price" is the price corresponding to the order w/ the least orders
            # if the popular sell price is above ours, make this our theo
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)



class RainforestResinStrategy(MarketMakingStrategy):
    def __init__(self, symbol, limit, buy_spread=1, sell_spread=1, buy_tolerance=0.5, sell_tolerance=0.5):
        super().__init__(symbol, limit, buy_spread, sell_spread, buy_tolerance, sell_tolerance)

    def get_true_value(self, state):
        return 10_000

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol, limit, buy_spread=1, sell_spread=2, buy_tolerance=0, sell_tolerance=0.5):
        super().__init__(symbol, limit, buy_spread, sell_spread, buy_tolerance, sell_tolerance)

    def get_true_value(self, state):
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((hit_buy_price + hit_sell_price) / 2)

class Trader:
    def __init__(self):
        limits = {
            "KELP":50,
            "RAINFOREST_RESIN":50
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        logger.print(state.position)

        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    