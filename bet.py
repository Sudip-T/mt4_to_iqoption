import os
import csv
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta



def get_martingale_sequence(
    capital: float, 
    levels: int = 7, 
    payout: float = 0.85, 
    precision: float = 0.01
) -> List:
    """
    Find the maximum initial bet that fits a Martingale sequence within given capital.

    Args:
        capital (float): Available capital
        levels (int): Number of martingale levels
        payout (float): Broker payout ratio
        precision (float): Binary search precision

    Returns:
        MartingaleResult: Optimal martingale calculation results

    Raises:
        ValueError: If input parameters are invalid
    """
    if capital <= 0:
        raise ValueError("Capital must be positive")
    if levels <= 0:
        raise ValueError("Levels must be positive")
    if not 0 < payout < 1:
        raise ValueError("Payout must be between 0 and 1")


    def simulate_martingale(initial_bet: float) -> Tuple[List[float], float]:
        bet_sequence = []
        current_bet = initial_bet
        cumulative_loss = 0

        for _ in range(levels):
            bet_sequence.append(round(current_bet, 2))
            cumulative_loss += current_bet
            current_bet = cumulative_loss / payout

        return bet_sequence, cumulative_loss

    # Binary search for optimal initial bet
    low = 0.01
    high = capital
    best_sequence = []
    best_total = 0

    while high - low > precision:
        mid = (low + high) / 2
        sequence, total = simulate_martingale(mid)

        if total <= capital:
            best_sequence = sequence
            best_total = total
            low = mid
        else:
            high = mid

    return best_sequence


class AutoTradingBot:
    def __init__(self, 
                 initial_balance: float,
                 stop_loss_amount: float,
                 daily_target: float,
                 martingle_steps:int = 7):

        self.initial_balance = initial_balance
        self.last_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_sl_limit = stop_loss_amount
        self.daily_tg_limit = daily_target
        
        # Martingale settings
        self.martingale_levels = martingle_steps
        self.martingale_multiplier = 2.0
        self.current_martingale_step = 0
        self.martingale_amounts = get_martingale_sequence(capital=self.initial_balance, levels=self.martingale_levels)
        
        # Tracking variables
        self.loss_today = 0
        self.profit_today = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_balance_for_martingale = initial_balance
        
        # CSV logging
        self.csv_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.initialize_csv()
        
        self.is_running = True
        self.stop_reason = None


    def initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'orderid_tn', 'symbol', 'signal', 'bet_amount', 'outcome', 'bl_before',
            'payout', 'bl_after', 'daily_profit',
            'martingale_step', 'consecutive_losses', 'stop_reason'
        ]
        
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def get_current_bet_amount(self) -> float:
        """Get the current bet amount based on martingale step"""
        if self.current_martingale_step >= self.martingale_levels:
            # If we exceed martingale steps, reset to step 0
            self.current_martingale_step = 0
        
        return self.martingale_amounts[self.current_martingale_step]
    
    def place_trade(self):
        return True, 1
    
    def check_outcome(self, bet_amt):

        return True, random.choice([-bet_amt, bet_amt * 1.85])

    def start_trading_session(self):
        """Run the trading session"""
        print(f"Starting trading session...")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Stop Loss: ${self.daily_sl_limit:.2f}")
        print(f"Daily Target: ${self.daily_tg_limit:.2f}")
        print(f"CSV Log: {self.csv_filename}")
        print("-" * 60)

        while self.is_running and self.total_trades < 5:
            # Check stop conditions before each trade
            if self.stop_reason != None:
                self.is_running = False
                print(self.stop_reason)
                return False, self.stop_reason

            bet_amount = self.get_current_bet_amount()
            if bet_amount > self.current_balance:
                if self.current_balance >= 1:
                    bet_amount = self.current_balance
                else:
                    self.stop_reason = "INSUFFICIENT_FUNDS"
                    return False, self.stop_reason
            
            placed, orderid = self.place_trade()
            if placed:
                self.current_balance = self.current_balance - bet_amount
                self.total_trades += 1
                closed, outcome = self.check_outcome(bet_amount)
            
            
            if closed and outcome > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
                self.consecutive_wins = 1
                self.current_martingale_step = 0
                self.profit_today += outcome
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.current_martingale_step += 1
                self.loss_today -= outcome

            self.current_balance += outcome

            self.log_trade(bet_amount, outcome, self.current_balance)
            
            # Check if we need to recalculate martingale levels
            if self.current_balance > self.last_balance:
                print(f"Balance increased from ${self.last_balance:.2f} to ${self.current_balance:.2f}")
                get_martingale_sequence(self.current_balance)

            time.sleep(1)
        
        self.print_final_summary()



    def print_final_summary(self):
        """Print final trading session summary"""
        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Stop Reason: {self.stop_reason}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total P/L: ${self.current_balance - self.initial_balance:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.total_trades-self.winning_trades}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        print(f"Max Consecutive Losses: {self.consecutive_losses}")
        print(f"Max Consecutive Wins: {self.consecutive_wins}")
        print(f"CSV Log File: {self.csv_filename}")
        print("="*60)

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'daily_pnl': self.current_balance - self.initial_balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': {self.total_trades-self.winning_trades},
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'consecutive_losses': self.consecutive_losses,
            'current_martingale_step': self.current_martingale_step + 1,
            'is_running': self.is_running,
            'stop_reason': self.stop_reason
        }
    
    def log_trade(self, bet_amount: float, 
                  outcome: bool, balance_after: float):
        """Log trade details to CSV"""
        trade_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.total_trades,
            f"{bet_amount:.2f}",
            "WIN" if outcome else "LOSS",
            f"{balance_after:.2f}",
            f"{self.profit_today:.2f}",
            self.current_martingale_step + 1,
            self.consecutive_losses,
            self.stop_reason or "ACTIVE"
        ]
        
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(trade_data)

if __name__ == "__main__":
    INITIAL_BALANCE = 100
    STOP_LOSS = 30
    DAILY_TARGET = 300
    MARTIGNLE_STEPS = 7
    
    bot = AutoTradingBot(
        daily_target=DAILY_TARGET,
        stop_loss_amount=STOP_LOSS,
        martingle_steps=MARTIGNLE_STEPS,
        initial_balance=INITIAL_BALANCE,
    )

    bot.start_trading_session()