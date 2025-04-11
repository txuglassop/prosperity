from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, TypeAlias
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
import json
import jsonpickle
import string
import math

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
    A generic trading class.

    Need to implement the act method for any instance of this class, which
    defines how it takes in TradingState and makes trades.

    In any instance of this class, use the `self.buy` and `self.sell` to send
    buy/sell orders when creating the `run` method.
    """
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        """
        Method responsible for executing the trades of the strategy. Attach trades to `self.trades`.

        :param state: Trading state object for the current timestamp
        """
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.position = state.position.get(self.symbol, 0)
        self.aggregate_buy_quantity = 0
        self.aggregate_sell_quantity = 0

        self.act(state)

        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        """
        Executes a BUY order for this product.
        Attempts to buy for the entire target quantity (without exceeding self.limit).

        :param price: Price level to use for the order
        :param quantity: Target quantity of the order (should be > 0)
        """
        if quantity <= 0:
            logger.print(f"Invalid call: buy(price={price}, quantity={quantity}). Quantity should be greater than 0.")
            return
        
        quantity = min(self.position + quantity, self.limit - self.aggregate_buy_quantity) - self.position
        if quantity == 0:
            return

        self.orders.append(Order(self.symbol, price, quantity))
        self.aggregate_buy_quantity += quantity

    def sell(self, price: int, quantity: int) -> None:
        """
        Executes a SELL order for this product.
        Attempts to sell for the entire target quantity (without exceeding self.limit).

        :param price: Price level to use for the order
        :param quantity: Target quantity of the order (should be > 0)
        """
        if quantity <= 0:
            logger.print(f"Invalid call: sell(price={price}, quantity={quantity}). Quantity should be greater than 0.")
            return
        
        quantity = self.position - max(self.position - quantity, -self.limit + self.aggregate_sell_quantity)
        if quantity == 0:
            return

        self.orders.append(Order(self.symbol, price, -quantity))
        self.aggregate_sell_quantity += quantity

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
        self.window_size = 10 # sensitivity to limit positions
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
            # the "popular price" is the price corresponding to the order w/ the most orders
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

class MarketMakingGLFTStrategy(Strategy):
    '''
    Guéant Lehalle Fernandez-Tapia Market Making Model
    https://hftbacktest.readthedocs.io/en/py-v2.0.0/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html
    '''
    def __init__(self, symbol: Symbol, limit: int, trade_volume: int, sigma: float, gamma: float, kappa: float, A: float, xi: float):
        """
        :param sigma: Price volatility
        :param gamma: Market impact parameter
        :param kappa: Market order arrival intensity
        :param A: Liquidity parameter
        :param xi: Inventory risk aversion
        """ 
        super().__init__(symbol, limit)
        self.trade_volume = trade_volume
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = kappa
        self.A = A
        self.xi = xi
    
    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        raise NotImplementedError()
    
    def act(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        
        q = self.position / self.limit

        sigma, gamma, kappa, A, xi = self.sigma, self.gamma, self.kappa, self.A, self.xi

        c1 = 1 / xi * math.log(1 + xi / kappa)
        c2 = sigma * math.sqrt(gamma / (2 * A * kappa) * (1 + xi / kappa)**(kappa / xi + 1))

        delta_b = c1 + 1 / 2 * c2 + q * c2
        delta_a = c1 + 1 / 2 * c2 - q * c2

        true_value = self.get_true_value(state)

        optimal_bid = round(true_value - delta_b)
        optimal_ask = round(true_value + delta_a)

        # Market take good ask prices
        if len(order_depth.sell_orders) != 0:
            for ask, quantity in sorted(order_depth.sell_orders.items(), key=lambda item: item[0]):
                if ask < optimal_ask:
                    self.buy(ask, -quantity)
                    
        # Market take good bid prices
        if len(order_depth.buy_orders) != 0:
            for bid, quantity in sorted(order_depth.buy_orders.items(), key=lambda item: -item[0]):
                if bid > optimal_bid:
                    self.sell(bid, quantity)

        # Market make
        self.buy(optimal_bid, self.trade_volume)
        self.sell(optimal_ask, self.trade_volume)

class KelpStrategy(MarketMakingGLFTStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit, trade_volume=16, sigma=0.4, gamma=0.65, kappa=2, A=0.1, xi=1)

    def get_true_value(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        vwa_ask = np.average(list(order_depth.sell_orders.keys()), weights=list(order_depth.sell_orders.values()))
        vwa_bid = np.average(list(order_depth.buy_orders.keys()), weights=list(order_depth.buy_orders.values()))
        return (vwa_ask + vwa_bid) / 2

# class KelpStrategy(MarketMakingStrategy):
#     def __init__(self, symbol, limit, buy_spread=1, sell_spread=1, buy_tolerance=0.5, sell_tolerance=0.5):
#         super().__init__(symbol, limit, buy_spread, sell_spread, buy_tolerance, sell_tolerance)

#     def get_true_value(self, state):
#         order_depth = state.order_depths[self.symbol]
#         buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
#         sell_orders = sorted(order_depth.sell_orders.items())

#         hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
#         hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

#         return round((hit_buy_price + hit_sell_price) / 2)


class History():
    def __init__(self):
        self.price = deque()

    def __len__(self):
        return len(self.price)

class SquidInkStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.history = History()

    def update_history(self, price):
        self.history.price.append(price) 
        if len(self.history.price) > 15: # history window
            self.history.price.pop()

    def get_weighted_average(self):
        values = np.array(self.history.price)
        weights = np.arange(1, len(self.history) + 1)
        return round(np.mean(values))
        #return round(np.average(values, weights=weights))

    def act(self, state: TradingState) -> None:
        base_value = 2000
        
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # update history
        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        avg_price = round((hit_buy_price + hit_sell_price) / 2)
        self.update_history(avg_price)
        
        if len(self.history.price) == 0:
            return

        # Get data from history
        mean = self.get_weighted_average()
        
        # find out how many items we can buy/sell
        position = state.position.get(self.symbol, 0)
        pos_window_size = 120
        pos_window_max_var = 153
        pos_window_center = self.limit*(base_value - mean)/pos_window_max_var
        pos_window_bottom = max(-self.limit, pos_window_center-pos_window_size/2)
        pos_window_top = min(self.limit, pos_window_center+pos_window_size/2)
        
        to_buy = max(pos_window_top - position,0)
        to_sell = max(-pos_window_bottom + position,0)
        
        #to_buy = self.limit - position
        #to_sell = self.limit + position

        prop_limit = position/self.limit
        if prop_limit >= 0:
            sell_limit_factor = max((1-prop_limit)**2,0.4)
            buy_limit_factor = 1 + sell_limit_factor
        else:
            buy_limit_factor = max((1+prop_limit)**2,0.4)
            sell_limit_factor = 1 + buy_limit_factor
        
        
        buy_buffer = 4
        buy_base_value_diff_factor = 2
        buy_weighting = 1+(buy_base_value_diff_factor*(hit_sell_price/base_value-1))
        
        # Smaller buy buffer means we buy more!!!
        # buy_weighting < 1 if hit sell price is below base_value (Buy more when price below base)
        # buy_limit_factor < 1 if we are negative position (Buy more when we are short)
        adj_buy_buffer = buy_buffer*buy_weighting*buy_limit_factor
        best_buy_price = round(mean - adj_buy_buffer)
        
        sell_buffer = 4
        sell_base_value_diff_factor = 2
        sell_weighting = 1-(sell_base_value_diff_factor*(hit_buy_price/base_value-1))
        
        # Smaller sell buffer means we sell more!!!
        # sell_weighting < 1 if hit sell price is above base_value (sell more when price above base)
        # sell_limit_factor < 1 if we are positive position (Sell more when we have shit to sell)
        adj_sell_buffer = sell_buffer*sell_weighting*sell_limit_factor
        best_sell_price = round(mean + adj_sell_buffer)
        
        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(best_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy) 

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(best_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell) 



class PicnicBasket1Strategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def find_avg_price(self, state: TradingState, symbol: str) -> int:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round(np.mean([hit_buy_price, hit_sell_price]))


    def act(self, state: TradingState):
        if any(symbol not in state.order_depths for symbol in ['CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']):
            return
        
        croissants = self.find_avg_price(state, 'CROISSANTS')
        djembes = self.find_avg_price(state, 'DJEMBES')
        jams = self.find_avg_price(state, 'JAMS')
        pb1 = self.find_avg_price(state, 'PICNIC_BASKET1')

        diff = pb1 - 6*croissants - 3*jams - djembes

        ### @ RAPH im not sure what this is for
        # but i used it for what i thought it was meant for
        # so yeah change it if i  was wrong
        buy_window = 50
        sell_window = 50
        
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        order_depth = state.order_depths[self.symbol]

        if diff >= sell_window:
            # basket is obervalued - we go short
            # take min price so we end up going as short as possible
            price = min(order_depth.buy_orders.keys())
            self.sell(price, to_sell)
        elif diff <= -buy_window:
            # basket is undervalued - we go long
            # take max price so we end up going as long as possible
            price = max(order_depth.sell_orders.keys())
            self.buy(price, to_buy)
        




    
class PicnicBasket2Strategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def find_avg_price(self, state: TradingState, symbol: str) -> int:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round(np.mean([hit_buy_price, hit_sell_price]))

    def act(self, state: TradingState):
        if any(symbol not in state.order_depths for symbol in ['CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']):
            return
        
        croissants = self.find_avg_price(state, 'CROISSANTS')
        jams = self.find_avg_price(state, 'JAMS')
        pb2 = self.find_avg_price(state, 'PICNIC_BASKET2')

        diff = pb2 - 4*croissants - 2*jams

        ### @ RAPH im not sure what this is for
        # but i used it for what i thought it was meant for
        # so yeah change it if i  was wrong
        buy_window = 50
        sell_window = 50
        
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        order_depth = state.order_depths[self.symbol]

        if diff >= sell_window:
            # basket is obervalued - we go short
            # take min price so we end up going as short as possible
            price = min(order_depth.buy_orders.keys())
            self.sell(price, to_sell)
        elif diff <= -buy_window:
            # basket is undervalued - we go long
            # take max price so we end up going as long as possible
            price = max(order_depth.sell_orders.keys())
            self.buy(price, to_buy)


class CroissantStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def find_avg_price(self, state: TradingState, symbol: str) -> int:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round(np.mean([hit_buy_price, hit_sell_price]))

    def act(self, state: TradingState):
        if any(symbol not in state.order_depths for symbol in ['CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']):
            return
        
        croissants = self.find_avg_price(state, 'CROISSANTS')
        djembes = self.find_avg_price(state, 'DJEMBES')
        jams = self.find_avg_price(state, 'JAMS')
        pb1 = self.find_avg_price(state, 'PICNIC_BASKET1')
        pb2 = self.find_avg_price(state, 'PICNIC_BASKET2')

        diff1 = pb1 - 6*croissants - 3*jams - djembes
        diff2 = pb2 - 4*croissants - 2*jams

        # if diff1 and diff2 imply different things about the underlying...
        # do absolutely fucking nothing!
        if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
            return
        
        # we only enter a position if both baskets indicate yes!
        buy_window = 50
        sell_window = 50
        
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        order_depth = state.order_depths[self.symbol]

        if diff1 >= buy_window and diff2 >= buy_window:
            # croissant is undervalued - we go long
            price = max(order_depth.buy_orders.keys())
            self.buy(price, to_buy)
        elif diff1 <= -sell_window and diff2 <= -sell_window:
            # croissant is overvalued - we go short
            price = min(order_depth.sell_orders.keys())
            self.sell(price, to_sell)
        



class JamStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def find_avg_price(self, state: TradingState, symbol: str) -> int:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round(np.mean([hit_buy_price, hit_sell_price]))

    def act(self, state: TradingState):
        croissants = self.find_avg_price(state, 'CROISSANTS')
        djembes = self.find_avg_price(state, 'DJEMBES')
        jams = self.find_avg_price(state, 'JAMS')
        pb1 = self.find_avg_price(state, 'PICNIC_BASKET1')
        pb2 = self.find_avg_price(state, 'PICNIC_BASKET2')

        diff1 = pb1 - 6*croissants - 3*jams - djembes
        diff2 = pb2 - 4*croissants - 2*jams

        # if diff1 and diff2 imply different things about the underlying...
        # do absolutely fucking nothing!
        if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
            return
        
        # we only enter a position if both baskets indicate yes!
        buy_window = 50
        sell_window = 50
        
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        order_depth = state.order_depths[self.symbol]

        if diff1 >= buy_window and diff2 >= buy_window:
            # jam is undervalued - we go long
            price = max(order_depth.buy_orders.keys())
            self.buy(price, to_buy)
        elif diff1 <= -sell_window and diff2 <= -sell_window:
            # jam is overvalued - we go short
            price = min(order_depth.sell_orders.keys())
            self.sell(price, to_sell)


class DjembeStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def find_avg_price(self, state: TradingState, symbol: str) -> int:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round(np.mean([hit_buy_price, hit_sell_price]))

    def act(self, state: TradingState):
        """
        Since Djembe is only connected to picnic basket 1, we only need to consider the
        difference between the underlying's of PB1 and the market price of PB1
        """
        if any(symbol not in state.order_depths for symbol in ['CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']):
            return
        
        croissants = self.find_avg_price(state, 'CROISSANTS')
        djembes = self.find_avg_price(state, 'DJEMBES')
        jams = self.find_avg_price(state, 'JAMS')
        pb1 = self.find_avg_price(state, 'PICNIC_BASKET1')

        diff = pb1 - 6*croissants - 3*jams - djembes

        ### @ RAPH im not sure what this is for
        # but i used it for what i thought it was meant for
        # so yeah change it if i  was wrong
        buy_window = 50
        sell_window = 50
        
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        order_depth = state.order_depths[self.symbol]

        if diff >= buy_window:
            # djembe is undervalued - we go long
            price = max(order_depth.buy_orders.keys())
            self.buy(price, to_buy)
        elif diff <= -sell_window:
            # djembe is overvalued - we go short
            price = min(order_depth.sell_orders.keys())
            self.sell(price, to_sell)



class Trader:
    def __init__(self):
        limits = {
            "KELP":50,
            "RAINFOREST_RESIN":50,
            "SQUID_INK":50,
            "CROISSANTS":250,
            "JAMS":350,
            "DJEMBES":60,
            "PICNIC_BASKET1":60,
            "PICNIC_BASKET2":100
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidInkStrategy,
            "CROISSANTS": CroissantStrategy,
            "JAMS": JamStrategy,
            "DJEMBES": DjembeStrategy,
            "PICNIC_BASKET1": PicnicBasket1Strategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy
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

