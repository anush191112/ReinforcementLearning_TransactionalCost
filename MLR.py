import pandas as pd
import numpy as np
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import SAC
import torch

class TradingEnv(gym.Env):
    def __init__(self, data, initial_inventory=1000,slippage_weight = 0.5, market_impact_weight = 0.5 ):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_inventory = initial_inventory
        self.current_step = 0
        self.remaining_inventory = initial_inventory
        
        # Define action and observation space
        # Action space: continuous range from 0 to remaining inventory (trade size)
        self.action_space = spaces.Box(low=0, high=self.remaining_inventory, shape=(1,), dtype=np.float32)
        
        # Observation space: volatility, volume, and inventory remaining
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Penalty weights for slippage and market impact
        self.slippage_weight = slippage_weight
        self.market_impact_weight = market_impact_weight
    
    def reset(self):
        self.current_step = 0
        self.remaining_inventory = self.initial_inventory
        return self._get_state()
    
    def step(self, action):
        # Action is the trade size in this minute
        trade_size = min(action[0], self.remaining_inventory)
        self.remaining_inventory -= trade_size

        # Calculate reward: simplified transaction cost minimization example
        current_price = self.data.iloc[self.current_step]['close']
        slippage_cost = self.slippage_weight * np.abs(trade_size * 0.01 * current_price)  # Example slippage cost
        market_impact_cost = self.market_impact_weight * np.abs(trade_size * 0.02 * current_price) # Example impact cost
        reward = -(slippage_cost+market_impact_cost)

        # Advance the environment
        self.current_step += 1
        done = self.current_step >= len(self.data) or self.remaining_inventory <= 0

        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # Current observation: normalized volatility, volume, and remaining inventory
        row = self.data.iloc[self.current_step]
        state = np.array([
            row['volatility'] / 100,
            row['volume'] / self.data['volume'].max(),
            self.remaining_inventory / self.initial_inventory
        ])
        return state
    
# Load your market data
data = pd.read_csv("/Users/anushreegupta/Downloads/merged_bid_ask_ohlcv_data.csv")  
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

data.drop(columns=['symbol'], inplace=True)


# Instantiate the environment with your data
env = TradingEnv(data)

# Define the SAC model
model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99, batch_size=256)

# Train the model
model.learn(total_timesteps=10000)  # Adjust timesteps for thorough training

# Save the model
model.save("sac_trading_model")

class Benchmark:
    
    def __init__(self, data):
        """
        Initializes the Benchmark class with provided market data.

        Parameters:
        data (DataFrame): A DataFrame containing market data, including top 5 bid prices and sizes.
        """
        self.data = data
        
    def get_twap_trades(data, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Time-Weighted Average Price (TWAP) strategy.

        Parameters:
        data (DataFrame): The input data containing timestamps and closing prices for each time step.
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the TWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        total_steps = len(data)
        twap_shares_per_step = initial_inventory / preferred_timeframe
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            size_of_slice = min(twap_shares_per_step, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['close'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)

    def get_vwap_trades(data, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Volume-Weighted Average Price (VWAP) strategy.

        Parameters:
        data (DataFrame): The input data containing timestamps, closing prices, and volumes for each time step.
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the VWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        total_volume = data['volume'].sum()
        total_steps = len(data)
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            volume_at_step = data['volume'].iloc[step]
            size_of_slice = (volume_at_step / total_volume) * initial_inventory
            size_of_slice = min(size_of_slice, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['close'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)
    
    def calculate_vwap(self, idx, shares):
        """
        Calculates the Volume-Weighted Average Price (VWAP) for a given step and share size.

        Parameters:
        idx (int): The index of the current step in the market data.
        shares (int): The number of shares being traded at the current step.

        Returns:
        float: The calculated VWAP price for the current step.
        """
        # Assumes you have best 5 bid prices and sizes in your dataset
        bid_prices = [self.data[f'bid_price_{i}'] for i in range(1, 6)]
        bid_sizes = [self.data[f'bid_size_{i}'] for i in range(1, 6)]
        cumsum = 0
        for idx, size in enumerate(bid_sizes):
            cumsum += size
            if cumsum >= shares:
                break
        
        return np.sum(bid_prices[:idx + 1] * bid_sizes[:idx + 1]) / np.sum(bid_sizes[:idx + 1])

    def compute_components(self, alpha, shares, idx):
        """
        Computes the transaction cost components such as slippage and market impact for a given trade.

        Parameters:
        alpha (float): A scaling factor for market impact (determined empirically or based on research).
        shares (int): The number of shares being traded at the current step.
        idx (int): The index of the current step in the market data.

        Returns:
        array: A NumPy array containing the slippage and market impact for the given trade.
        """
        actual_price = self.calculate_vwap(idx, shares)
        Slippage = (self.data['bid_price_1'] - actual_price) * shares  # Assumes bid_price_1 is in your dataset
        Market_Impact = alpha * np.sqrt(shares)
        return np.array([Slippage, Market_Impact])
    
    def simulate_strategy(trades, ohlcv_data, preferred_timeframe):
        slippage = []
        market_impact = []

        for index, row in trades.iterrows():
            trade_price = row['price']
            trade_shares = row['shares']

            # Check if there's enough data to evaluate slippage and market impact
            # Ensure you are correctly indexing into ohlcv_data
            if index < len(ohlcv_data):
                market_price = ohlcv_data.iloc[index]['close']

                # Calculate slippage
                slippage_amount = (trade_price - market_price) * trade_shares
                slippage.append(slippage_amount)

                # Calculate market impact (for example purposes)
                impact_amount = trade_shares * (trade_price - market_price) / market_price
                market_impact.append(impact_amount)

        return slippage, market_impact
    

class Rl_benchmark:
    def __init__(self, data):
        """Initializes the Rl_benchmark class with provided market data."""
        self.data = data

    def calculate_trade_sizes(self, total_shares, time_horizon=390):
        """Calculates trade sizes dynamically for RL model."""
        trade_sizes = []
        remaining_shares = total_shares

        for minute in range(time_horizon):
            trade_size = remaining_shares / (time_horizon - minute)
            remaining_shares -= trade_size
            trade_sizes.append(trade_size)
        
        return trade_sizes

    @staticmethod
    def get_initial_state(data):
        """Get the initial state for the RL model."""
        state = np.array([data['volatility'].iloc[0] / 100,
                          data['volume'].iloc[0] / data['volume'].max(),
                          1.0])  # Normalized remaining inventory
        return state 

    @staticmethod
    def get_next_state(data, current_step, remaining_inventory):
        """Calculate the next state based on current step data and remaining inventory."""
        row = data.iloc[current_step]
        next_state = np.array([
            row['volatility'] / 100,
            row['volume'] / data['volume'].max(),
            remaining_inventory / 1000  # Normalized to initial inventory
        ])
        return next_state

    @staticmethod
    def get_rl_trades(model, data, initial_inventory, preferred_timeframe=390):
        """Generate trade schedule using the RL model."""
        total_steps = len(data)
        remaining_inventory = initial_inventory
        trades = []

        # Initialize state
        state = Rl_benchmark.get_initial_state(data)
        
        for step in range(min(total_steps, preferred_timeframe)):
            with torch.no_grad():
                # Ensure state is reshaped correctly for model prediction
                action, _ = model.predict(state, deterministic=True)
            
            # Calculate trade size and update inventory
            trade_size = min(action[0], remaining_inventory)
            remaining_inventory -= int(np.ceil(trade_size))
            
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['close'],
                'shares': trade_size,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
            
            # Update the state for the next step
            state = Rl_benchmark.get_next_state(data, step, remaining_inventory)

            if remaining_inventory <= 0:
                break

        return pd.DataFrame(trades)


# Set up parameters
initial_inventory = 1000
timeframe = 390  # Full trading day
    
# Run TWAP
twap_schedule = Benchmark.get_twap_trades(data, initial_inventory, timeframe)
print("TWAP Schedule:\n", twap_schedule.head())

# Run VWAP
vwap_schedule = Benchmark.get_vwap_trades(data, initial_inventory, timeframe)
print("VWAP Schedule:\n", vwap_schedule.head())

# Run RL-Based Schedule (requires a trained SAC model)
model = SAC.load("sac_trading_model")
rl_schedule = Rl_benchmark.get_rl_trades(model, data, initial_inventory, timeframe)
print("RL-Based Schedule:\n", rl_schedule.head())

# Define a function to calculate transactional costs based on trade schedule and market data
def calculate_transaction_costs(trade_df, market_df):
    # Merge trade data with market data on timestamp
    df = pd.merge(trade_df, market_df[['timestamp', 'bid_price_1', 'ask_price_1']], on='timestamp', how='left')

    # Calculate mid-price and spread cost for each trade
    def cost_calculator(row):
        spread = row['ask_price_1'] - row['bid_price_1']
        mid_price = (row['bid_price_1'] + row['ask_price_1']) / 2
        executed_price = row['price']
        
        # Spread Cost: half of the spread times the number of shares traded
        spread_cost = spread / 2 * row['shares']
        
        # Market Impact: absolute deviation from mid-price
        market_impact = abs(executed_price - mid_price) * row['shares']
        
        # Slippage: absolute deviation from the mid-price, considered as a benchmark here
        slippage = abs(executed_price - mid_price) * row['shares']
        
        return pd.Series({
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'slippage': slippage,
            'total_cost': spread_cost + market_impact + slippage
        })

    # Apply cost calculation function to each row in the DataFrame
    df[['spread_cost', 'market_impact', 'slippage', 'total_cost']] = df.apply(cost_calculator, axis=1)
    
    # Return the DataFrame with calculated costs
    return df

# Apply the function to each trade schedule DataFrame
twap_costs = calculate_transaction_costs(twap_schedule, data)
vwap_costs = calculate_transaction_costs(vwap_schedule, data)
rl_costs = calculate_transaction_costs(rl_schedule, data)

# Sum costs for each method
twap_total_cost = twap_costs[['spread_cost', 'market_impact', 'slippage', 'total_cost']].sum()
vwap_total_cost = vwap_costs[['spread_cost', 'market_impact', 'slippage', 'total_cost']].sum()
rl_total_cost = rl_costs[['spread_cost', 'market_impact', 'slippage', 'total_cost']].sum()

# Create a summary DataFrame to compare costs across methods
cost_summary = pd.DataFrame({
    'Method': ['TWAP', 'VWAP', 'RL'],
    'Spread Cost': [twap_total_cost['spread_cost'], vwap_total_cost['spread_cost'], rl_total_cost['spread_cost']],
    'Market Impact': [twap_total_cost['market_impact'], vwap_total_cost['market_impact'], rl_total_cost['market_impact']],
    'Slippage': [twap_total_cost['slippage'], vwap_total_cost['slippage'], rl_total_cost['slippage']],
    'Total Cost': [twap_total_cost['total_cost'], vwap_total_cost['total_cost'], rl_total_cost['total_cost']]
})

print("Cost Summary by Method:")
print(cost_summary)


# Convert rl_schedule to JSON format
rl_schedule_json = rl_schedule.to_json(orient="records", date_format="iso")

# Save the JSON to a file
with open("rl_trade_schedule.json", "w") as json_file:
    json_file.write(rl_schedule_json)