import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import datetime
from scipy.optimize import minimize
from scipy.optimize import newton
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import brentq
from scipy.optimize import root_scalar
from scipy.optimize import fsolve
import sys
import os
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# helpers from fang_cmds

# p = Cz functions
def calc_discount_factor(quotes, price_col_name, cashflow_matrix): #z=C^-1p
    prices = quotes[price_col_name].to_numpy().reshape(-1, 1)
    cashflow_matrix = cashflow_matrix.to_numpy()

    cashflow_matrix_inverse = np.linalg.inv(cashflow_matrix)
    discount_factors = np.dot(cashflow_matrix_inverse,prices).flatten()

    return pd.DataFrame({'ttm': quotes['ttm'], 'discount factor': discount_factors}) # FINAL

def calc_prices(discount_data, discount_factor_col_name, cashflow_matrix): # p=Cz
    """
    Calculate bond prices using discount factors and a cash flow matrix.

    Parameters:
    - discount_data (pd.DataFrame): DataFrame containing discount factors.
    - discount_factor_col_name (str): Column name for discount factors in the discount_data DataFrame.
    - cashflow_matrix (pd.DataFrame): DataFrame containing the cash flow matrix.

    Returns:
    - tuple: A tuple containing:
        - np.ndarray: Array of calculated prices.
        - pd.DataFrame: DataFrame with time-to-maturity (ttm) and calculated prices.
    """
    discount_factors_np = discount_data[discount_factor_col_name].to_numpy().reshape(-1, 1)
    cashflow_matrix_np = cashflow_matrix.to_numpy()

    prices = np.dot(cashflow_matrix_np, discount_factors_np).flatten()
    prices_df = pd.DataFrame({'price': prices}, index = cashflow_matrix.index)
    
    return prices_df # FINAL

# duration
def calc_bond_duration(coupon_rate, ytm, frequency, years_to_maturity): # closed formula
    """
    Calculate the duration of a bond using the closed-form formula.

    Parameters:
    - coupon_rate (float): Annual coupon rate as a decimal (e.g., 0.05 for 5%).
    - ytm (float): Annual yield to maturity as a decimal (e.g., 0.04 for 4%).
    - frequency (int): Coupon payment frequency per year (e.g., 2 for semi-annual).
    - years_to_maturity (float): Time to maturity in years.

    Returns:
    - float: Duration of the bond.
    """
    # Calculate periodic yield (y_tilde), periodic coupon rate (c_tilde), and effective periods (tau_tilde)
    y_tilde = ytm / frequency
    c_tilde = coupon_rate / frequency
    tau_tilde = frequency * years_to_maturity

    # Compute the closed-form duration formula
    numerator = (1 + y_tilde) / y_tilde - (1 + y_tilde + tau_tilde * (c_tilde - y_tilde)) / (
        c_tilde * ((1 + y_tilde) ** tau_tilde - 1) + y_tilde
    )
    duration = (1 / frequency) * numerator

    return duration

# price bond using yield
def price_bond_using_yield(face_value, coupon_rate, ytm, coupon_frequency, time_to_maturity):
    """
    Calculate the price of a bond.

    Parameters:
    - face_value (float): The bond's face value (e.g., $1000).
    - coupon_rate (float): The annual coupon rate as a decimal (e.g., 0.05 for 5%).
    - ytm (float): The annual yield to maturity as a decimal (e.g., 0.04 for 4%).
    - coupon_frequency (int): Number of coupon payments per year (e.g., 2 for semi-annual).
    - time_to_maturity (float): Time to maturity in years.

    Returns:
    - float: The price of the bond.
    """
    # Calculate coupon payment
    coupon_payment = face_value * coupon_rate / coupon_frequency
    total_periods = int(time_to_maturity * coupon_frequency)
    periodic_ytm = ytm / coupon_frequency

    # Calculate the present value of coupon payments
    pv_coupons = sum(coupon_payment / ((1 + periodic_ytm) ** t) for t in range(1, total_periods + 1))

    # Calculate the present value of the face value
    pv_face_value = face_value / ((1 + periodic_ytm) ** total_periods)

    # Bond price is the sum of the two present values
    bond_price = pv_coupons + pv_face_value
    return bond_price

def calc_bond_ytm(price, coupon_rate, tenor, face_value=100, coupon_freq=2):
    """
    Calculate the yield to maturity (YTM) of a bond.

    Parameters:
    price (float): The current price of the bond.
    coupon_rate (float): The annual coupon rate of the bond as a decimal.
    tenor (float): The time to maturity of the bond in years.
    face_value (float, optional): The face value of the bond. Defaults to 100.
    coupon_freq (int, optional): The number of coupon payments per year. Defaults to 2 (semiannual).

    Returns:
    float: The calculated yield to maturity (YTM) of the bond. Returns NaN if the method fails.
    """
    # Compute the periodic coupon payment
    periods = int(tenor * coupon_freq)  # Total payment periods
    coupon_payment = (coupon_rate / coupon_freq) * face_value  # Periodic coupon
    
    # Define the function to solve for YTM
    def ytm_function(y):
        return sum(coupon_payment / (1 + y/coupon_freq) ** t for t in range(1, periods + 1)) + \
               face_value / (1 + y/coupon_freq) ** periods - price
    
    # Use Newton's method to solve for YTM
    try:
        ytm = newton(ytm_function, x0=0.02)  # Initial guess of 2%
    except RuntimeError:
        ytm = np.nan  # Return NaN if the method fails
    
    return ytm  # Convert periodic yield to annualized YTM

# convert from spot rate --> discount factor --> forward factor --> forward rate

def discount_rate_to_factor(df, ttm_col, discount_rate_col, new_discount_col, compounding_frequency):
    """
    Convert discount rates to discount factors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing time-to-maturity (TTM) and discount rates.
    - ttm_col (str): Column name for time-to-maturity (TTM) in years.
    - discount_rate_col (str): Column name for discount rates.
    - compounding_frequency (int): Compounding frequency (e.g., 1 for annual, 2 for semi-annual, etc.). Use 0 for continuous compounding.

    Returns:
    - pd.DataFrame: DataFrame with an added column 'discount_factor' containing calculated factors.
    """
    if ttm_col not in df.columns or discount_rate_col not in df.columns:
        raise ValueError("Specified columns not found in DataFrame.")

    df[ttm_col] = pd.to_numeric(df[ttm_col], errors='coerce')
    df[discount_rate_col] = pd.to_numeric(df[discount_rate_col], errors='coerce')

    if compounding_frequency == 0:
        # Continuous compounding
        df[new_discount_col] = np.exp(-df[discount_rate_col] * df[ttm_col])
    else:
        # Discrete compounding
        df[new_discount_col] = (1 + df[discount_rate_col] / compounding_frequency) ** (-compounding_frequency * df[ttm_col])

    return df

def spot_to_forward_factor(df, spot_discount_factor_col, forward_discount_factor_col):
    """
    Convert spot discount factors to forward discount factors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing forward discount factors.
    - spot_discount_factor_col (str): Column name for spot discount factors.
    - forward_discount_factor_col (str): Column name for forward discount factors.

    Returns:
    - pd.DataFrame: DataFrame with an added column containing calculated forward discount factors.
    """
    df[forward_discount_factor_col] = df[spot_discount_factor_col] / df[spot_discount_factor_col].shift(1)
    df[forward_discount_factor_col].iloc[0] = df[spot_discount_factor_col].iloc[0]
    return df

def forward_discount_factor_to_rate(df, delta, forward_discount_factor_col, new_rate_col, compounding_frequency=0):
    """
    Convert forward discount factors to forward discount rates.

    Parameters:
    - df (pd.DataFrame): DataFrame containing forward discount factors.
    - delta (float): Time difference between T1 and T2 (T2 - T1).
    - discount_factor_col (str): Column name for forward discount factors.
    - new_rate_col (str): Column name for the calculated forward discount rates.
    - compounding_frequency (int): Compounding frequency (e.g., 1 for annual, 2 for semi-annual, etc.).
        Use 0 for continuous compounding (default).

    Returns:
    - pd.DataFrame: DataFrame with an added column containing calculated forward discount rates.
    """
    if forward_discount_factor_col not in df.columns:
        raise ValueError("Specified discount factor column not found in DataFrame.")

    if compounding_frequency == 0:
        # Continuous compounding
        df[new_rate_col] = -np.log(df[forward_discount_factor_col]) / delta
    else:
        # Discrete compounding
        df[new_rate_col] = compounding_frequency * (
            (df[forward_discount_factor_col] ** (-1 / (compounding_frequency * delta))) - 1
        )

    return df

###### START OF NEW MATERIAL

#  blacks formula
def blacks_formula(T,vol,strike,fwd,discount=1,isCall=True):
    """
    Calculate the price of a European option using Black's formula.

    Parameters:
    T (float): Time to maturity in years.
    vol (float): Volatility as a decimal.
    strike (float): Strike rate.
    fwd (float): Forward rate.
    discount (float, optional): Discount factor. Defaults to 1.
    isCall (bool, optional): True if the option is a call, False if it is a put. Defaults to True.

    Returns:
    float: Price of the option.
    """
    sigT = vol * np.sqrt(T)
    d1 = (1/sigT) * np.log(fwd/strike) + .5*sigT
    d2 = d1-sigT
    
    if isCall:
        val = discount * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        val = discount * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return val

# CALLABLE BOND

# 1. price the forward bond
def calc_forward_bond_price(price_df, cf_mat_mod, discs, quote_data, bond_name):
    # calculate time until next callable date (time today to T_option)
    callable_date = quote_data.loc[bond_name, 'Date of First Possible Call']
    quote_date = quote_data.loc[bond_name, 'quote date']
    option_ttm_rounded = round((callable_date - quote_date).days / 365,1)
    
    # P(T)
    hypothetical_price = price_df.loc[bond_name, 'price']
    # PV of Coupons
    coupons = cf_mat_mod.loc[bond_name, :option_ttm_rounded]
    discount_factors = discs.loc[:option_ttm_rounded, 'discount']
    sum_discounted_coupons = np.dot(coupons, discount_factors)
    # Forward Bond Price
    numerator = (hypothetical_price - sum_discounted_coupons)
    denominator = 100 * discs.loc[option_ttm_rounded, 'discount']
    forward_bond_price = numerator / denominator
    
    return forward_bond_price

def mark_calc_forward_bond_price(spot,Tfwd,discount_curve,cpnrate,face=100,cpnfreq=2):
    """
    Calculate the forward price of a bond.

    Parameters:
    spot (float): The current spot price of the bond.
    Tfwd (float): The forward time in years.
    discount_curve (pd.DataFrame): DataFrame containing the discount factors indexed by time-to-maturity.
    cpnrate (float): The annual coupon rate of the bond.
    face (float, optional): The face value of the bond. Defaults to 100.
    cpnfreq (int, optional): The number of coupon payments per year. Defaults to 2 (semiannual).

    Returns:
    float: The forward price of the bond.
    """
    discount_grid_step = np.diff(discount_curve.index).mean()
    grid_step_cpn = round(1 / (cpnfreq * discount_grid_step))
    Tfwd_rounded = get_approximate_discount(Tfwd,discount_curve)

    Z = discount_curve.loc[Tfwd_rounded,'discount']
    cpn_discs = discount_curve.loc[:Tfwd_rounded:grid_step_cpn,'discount']

    coupon_payment = face * cpnrate / cpnfreq
    pv_coupons = sum(coupon_payment * df for df in cpn_discs)
    fwd_price = (spot - pv_coupons) / Z

    return fwd_price

def get_approximate_discount(T,discs):
    diffs_array = np.abs(discs.index - T)
    imin = diffs_array.argmin()
    idx = discs.index[imin] 
    return idx

# 2. calc the implied vol
def ratevol_to_pricevol(ratevol,rate,duration):
    """
    Convert interest rate volatility to price volatility.

    Parameters:
    ratevol (float): The volatility of the interest rate.
    rate (float): The interest rate.
    duration (float): The duration of the bond.

    Returns:
    float: The price volatility of the bond.
    """
    pricevol = ratevol * rate * duration
    return pricevol

# 3. apply black's formula to price the call
#  above
#  Inputs:
    # F_t = forward_bond_price # Forward bond price
    # K = 100  # Strike price (call price of the bond)
    # sigma = bond_iv # Bond implied volatility from ratevol_to_pricevol
    # T = option_ttm # Time to maturity in years -- option expiry
    # Z_t_T = discs.loc[option_ttm_rounded, 'discount'] # Discount factor

# 4. price the vanilla bond
#  use above functions (either price bond with yield or spot rates/discount factors)

# 5. price the callable bond
def calc_callable_price(vanilla_bond_price, call_price):
    return vanilla_bond_price - call_price

# wrapper for callable bond price
def calc_callable_bond_price(face_value, coupon_rate, ytm, coupon_frequency, time_to_maturity, T_option, vol, strike, fwd, discount, ratevol, rate, discount_curve):
    """
    Wrapper function to calculate the price of a callable bond.

    Parameters:
    face_value (float): The bond's face value (e.g., $1000).
    coupon_rate (float): The annual coupon rate as a decimal (e.g., 0.05 for 5%).
    ytm (float): The annual yield to maturity as a decimal (e.g., 0.04 for 4%).
    coupon_frequency (int): Number of coupon payments per year (e.g., 2 for semi-annual).
    time_to_maturity (float): Time to maturity in years.
    T_option (float): Time to the option expiry in years.
    vol (float): Volatility of the bond price.
    strike (float): Strike price of the call option.
    fwd (float): Forward rate.
    discount (float): Discount factor.
    ratevol (float): Volatility of the interest rate.
    rate (float): The interest rate.
    discount_curve (pd.DataFrame): DataFrame containing the discount factors indexed by time-to-maturity.

    Returns:
    float: The price of the callable bond.
    """
    vanilla_bond_price = price_bond_using_yield(face_value, coupon_rate, ytm, coupon_frequency, time_to_maturity)
    fwd = mark_calc_forward_bond_price(vanilla_bond_price,T_option,discount_curve,coupon_rate,face=100,cpnfreq=2)
    duration = calc_bond_duration(coupon_rate, ytm, coupon_frequency, time_to_maturity)
    vol = ratevol_to_pricevol(ratevol,rate,duration)
    call_price = blacks_formula(T_option, vol, strike, fwd, discount, isCall=True)
    return vanilla_bond_price - call_price

# calculate OAS
"""
# def bond_model_price(cf_mat_mod, quote_data, bond_name, spot_rates, coupon_rate, tenor, face_value=100, coupon_freq=2, spread=0):
#     
#     Computes the price of a bond given a spread over the risk-free rate.

#     Parameters:
#     spot_rates : pandas.Series
#         Spot rates indexed by time-to-maturity.
#     coupon_rate : float
#         Coupon rate as a decimal.
#     tenor : float
#         Time to maturity (years).
#     face_value : float, optional
#         Face value of the bond.
#     coupon_freq : int, optional
#         Number of coupon payments per year (default = 2 for semiannual).
#     spread : float, optional
#         Parallel shift in spot rates (as a decimal, e.g., 0.001 for 10 bps).

#     Returns:
#     float
#         Modeled price of the bond with the given spread.
#     
#     #print('spread:', spread)
#     cashflows = cf_mat_mod.loc[bond_name, :].values
#     #print(cashflows)
#     # PRICING HYPOTHETICAL BOND WITH NEW RATES
#     # Compute time steps for cash flows
#     times = np.arange(1 / coupon_freq, tenor + 1 / coupon_freq, 1 / coupon_freq)
#     #print(times)
#     # Adjust spot rates by adding the spread
#     #spot_rates_adjusted = np.interp(times, spot_rates.index.values, spot_rates.values) + spread
#     spot_rates_adjusted = spot_rates.values + spread
#     #print(spot_rates_adjusted)
#     # Convert to discount factors
#     discount_factors = np.exp(-spot_rates_adjusted * times)
#     #discount_factors = discs['discount'].loc[:bond_ttm_rounded].to_list()
#     #print(discount_factors)
#     #print(discount_factors)
#     # Compute coupon payments
#     #coupon_payment = (coupon_rate / coupon_freq) * face_value
#     # Compute bond price
#     hypothetical_price = np.dot(cashflows,discount_factors)
#     #print('hyp price:', hypothetical_price)

#     def calc_forward_bond_price(hypothetical_price, cashflows, discount_factors, quote_data, bond_name):
#         # calculate time until next callable date (time today to T_option)
#         callable_date = quote_data.loc[bond_name, 'Date of First Possible Call']
#         quote_date = quote_data.loc[bond_name, 'quote date']
#         option_ttm_rounded = round((callable_date - quote_date).days / 365,1)
        
#         # PV of Coupons
#         option_ttm_rounded = int(option_ttm_rounded)
#         coupons = cashflows[:option_ttm_rounded*2]
#         #print(coupons)
#         coupon_discount_factors = discount_factors[:int(option_ttm_rounded*2)]
#         sum_discounted_coupons = np.dot(coupons, coupon_discount_factors)
#         # Forward Bond Price
#         numerator = (hypothetical_price - sum_discounted_coupons)
#         denominator = 100 * discount_factors[int(option_ttm_rounded*2 - 1)]
#         forward_bond_price = numerator / denominator
        
#         return forward_bond_price

#     forward_price = calc_forward_bond_price(hypothetical_price, cashflows, discount_factors, quote_data, bond_name) * 100
#     #print('forward price:', forward_price)

#     bond_iv = quote_data.loc[bond_name, 'Duration'] * quote_data.loc[bond_name, 'Implied Vol'] * spot_rates_adjusted[int(option_ttm_rounded * 2 - 1)] / 100
#     #print('bond iv:', bond_iv)

#     # Use Black
#     option_price = blacks_formula(forward_price, bond_iv, option_ttm, 100, discount_factors[int(option_ttm_rounded * 2 -1)])
#     #print('option price:', option_price)

#     # Compute the callable bond price
#     callable_bond_price = hypothetical_price - option_price
#     #print('callable bond price:', callable_bond_price)

#     return callable_bond_price
"""

def calculate_oas(cf_mat_mod, quote_data, bond_name, spot_rates, price_market, price_model, coupon_rate, tenor, face_value=100, coupon_freq=2):
    """
    Computes the Option-Adjusted Spread (OAS) by solving for the spread that equates
    the modeled price to the market price.

    Parameters:
    spot_rates : pandas.Series
        Spot rates indexed by time-to-maturity.
    price_market : float
        Market price of the callable bond.
    price_model : float
        Modeled price of the bond before adding the spread.
    coupon_rate : float
        Coupon rate as a decimal.
    tenor : float
        Time to maturity (years).
    face_value : float, optional
        Face value of the bond.
    coupon_freq : int, optional
        Number of coupon payments per year (default = 2 for semiannual).

    Returns:
    float
        The OAS (spread in basis points).
    """

    # Solve for OAS where modeled price matches market price
    def oas_function(spread):
        return bond_model_price(cf_mat_mod, quote_data, bond_name, spot_rates, coupon_rate, tenor, face_value, coupon_freq, spread) - price_market

    # Use Brent's method to find OAS
    oas = brentq(oas_function, -0.005, 0.005)  # Searching for spread in range [-200bps, +200bps]

    return oas * 10000  # Convert to basis points (bps)

################################################################################

# CAPS/FLOORS

# 1. price a caplet/floorlet
def price_caplet_floorlet(F, K, DF_T, flat_vol, T, notional=100, frequency=4, isCaplet=True):
    """
    Price a caplet or floorlet using Black's formula.
    """
    step_size = 1/frequency
    option_price = blacks_formula(T-step_size, flat_vol, K, F, DF_T, isCall=isCaplet)
    return (notional / frequency) * option_price

def price_caplet_floorlet_draft(F, K, DF_T, sigma, T, notional=100, tau=0.25, isCaplet=True):
    """
    Price a caplet or floorlet using Black's formula.

    Parameters:
    - F: Forward rate (quarterly compounding)
    - K: Strike rate (swap rate at T=3)
    - DF_T: Discount factor to T=3
    - sigma: Volatility of forward rate
    - T: Time to expiry (years)
    - notional: Notional amount
    - tau: Time period per settlement (quarterly = 0.25)
    - isCaplet: Boolean flag (True for caplet, False for floorlet)

    Returns:
    - Caplet or floorlet price
    """
    # Compute d1 and d2
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if isCaplet:
        price = DF_T * notional * tau * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = DF_T * notional * tau * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return price

# F = rate_data.loc[3.0, 'forwards']
# K = rate_data.loc[3.0, 'swap rates']
# DF_T = rate_data.loc[3.0, 'discounts']
# sigma = rate_data.loc[3.0, 'flat vols']
# T = 2.75
# floorlet_price = price_floorlet(F, K, DF_T, sigma, T)
# print(floorlet_price)


# 2. price a cap/floor
def price_cap_floor(start_mat, end_mat, F, K, DF_T, flat_vol, step_size=0.25, isCap=True):
    """
    Price a cap or floor using a summation of caplets or floorlets.

    Parameters:
    - start_mat (float): Start maturity in years.
    - end_mat (float): End maturity in years.
    - F (float): Forward rate.
    - K (float): Strike rate.
    - DF_T (float): Discount factor to the maturity.
    - sigma (float): Volatility of the forward rate.
    - T (float): Time to expiry in years.
    - step_size (float, optional): Step size for periods (default quarterly = 0.25).
    - isCap (bool, optional): Boolean flag (True for cap, False for floor). Defaults to True.

    Returns:
    - float: Cap price or floor price.
    """
    times = np.arange(start_mat, end_mat + step_size, step_size)
    total_price = 0
    
    for time in times:
        price = price_caplet_floorlet(F, K, DF_T, flat_vol, time, notional=100, frequency=4, isCaplet=isCap)
        total_price += price
    
    return total_price

# 3. 

def generate_cap_prices(start_mat, end_mat, F, K, DF_T, sigma, T, step_size=0.25):
    """
    Generate cap prices for a range of maturities.

    Parameters:
    - start_mat (float): Start maturity in years.
    - end_mat (float): End maturity in years.
    - step_size (float): Step size for periods (e.g., 0.25 for quarterly).

    Returns:
    - dict: A dictionary where the keys are the maturities and the values are the corresponding cap prices.
    """
    cap_prices = {}
    times = np.arange(start_mat, end_mat+step_size, step_size)
    for ttm in times:
        cap_price = price_cap_floor(start_mat, end_mat, F, K, DF_T, sigma, T, step_size=0.25, isCap=True)
        cap_prices[ttm] = cap_price
    return cap_prices

# cap_prices = generate_cap_prices(0.5, 10.0, 0.25)
# cap_prices_df = pd.DataFrame(cap_prices, index=['cap price']).T
# cap_prices_df
################################################################################
# STRIP CAPS TO TO GET FORWARD VOLS

# price a cap using flat vol (MARK'S)
def cap_vol_to_price(flatvol, strike, fwds, discounts, dt=0.25, notional=100):
    """
    Computes the price of an interest rate cap using flat volatility and Black's formula.

    Parameters:
    - flatvol: float
        The flat implied volatility of the cap.
    - strike: float
        The strike rate of the cap.
    - fwds: pandas.Series
        A series of forward rates indexed by time.
    - discounts: pandas.Series
        A series of discount factors indexed by time.
    - dt: float, optional (default=0.25)
        The time interval in years between caplets (e.g., 0.25 for quarterly settlement).
    - notional: float, optional (default=100)
        The notional amount of the cap.

    Returns:
    - capvalue: float
        The total price of the cap, computed as the sum of individual caplet prices.

    Notes:
    - The function iterates through the time periods, applying Black's formula to each caplet.
    - The `blacks_formula` function is expected to compute the price of a single caplet.
    - The sum of all caplet values gives the overall cap price.
    """
    T = discounts.index[-1]
    flatvalues = pd.Series(dtype=float, index=discounts.index, name='flat values')

    tprev = discounts.index[0]
    for t in discounts.index[1:]:
        flatvalues.loc[t] = notional * dt * fv.blacks_formula(tprev, flatvol, strike, fwds.loc[t], discounts.loc[t])
        tprev = t

    capvalue = flatvalues.sum()
    return capvalue

# flatvol = rate_data.loc[0.5, 'flat vols']
# strike = rate_data.loc[0.5, 'swap rates']
# fwds = rate_data['forwards']
# discounts = rate_data['discounts']
# cap_vol_to_price(flatvol, strike, fwds, discounts, dt=0.25, notional=100)

# extract implied vol from caplet price (not actually used)
def implied_vol(caplet_price, F, K, DF_T, T, notional=100, tau=0.25, tol=1e-6):
    """
    Solves for the implied volatility given the market price of a caplet.

    Parameters:
    - caplet_price: Observed market price of the caplet
    - F: Forward rate (quarterly compounding)
    - K: Strike rate (swap rate at T)
    - DF_T: Discount factor to T
    - T: Time to expiry (years)
    - notional: Notional amount
    - tau: Time period per settlement (quarterly = 0.25)
    - tol: Tolerance for numerical root-finding

    Returns:
    - Implied volatility (sigma)
    """

    def black_caplet_price(sigma):
        """Black's formula for a caplet given volatility."""
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return DF_T * notional * tau * (F * norm.cdf(d1) - K * norm.cdf(d2))

    def objective(sigma):
        """Objective function: difference between market and model price."""
        return black_caplet_price(sigma) - caplet_price

    # Initial guesses: start with a reasonable range for sigma
    sigma_initial = 0.2  # 20% as a common starting point
    sigma_lower = 1e-6  # Near-zero lower bound
    sigma_upper = 2.0  # High bound to ensure convergence

    # Solve for implied volatility
    result = root_scalar(objective, bracket=[sigma_lower, sigma_upper], method='brentq', xtol=tol)

    if result.converged:
        return result.root
    else:
        raise ValueError("Implied volatility solver did not converge")

# ACTUAL FUNCTION TO EXTRACT FORWARD VOLATILITIES
def flat_to_forward_vol_rev(flatvols, strikes, fwds, discounts, freq=None, notional=100, returnCaplets=False):
    """
    Converts flat implied volatilities of caps into forward volatilities for caplets of different maturities.

    This function extracts the forward volatilities from market-observed flat volatilities quoted on interest rate caps.
    It follows a backward induction approach, solving for the forward volatilities that match the caplet prices
    implied by the given flat volatility. 

    Parameters:
    - flatvols: pandas.Series
        Series of flat volatilities (Black volatilities) for caps at different maturities.
    - strikes: pandas.Series
        Series of strike rates for the caps at different maturities.
    - fwds: pandas.Series
        Forward rate curve indexed by time.
    - discounts: pandas.Series
        Discount factor curve indexed by time.
    - freq: int, optional (default=None)
        Frequency of rate resets per year (e.g., 4 for quarterly).
    - notional: float, optional (default=100)
        Notional amount for the cap.
    - returnCaplets: bool, optional (default=False)
        If True, returns both forward volatilities and caplet prices.

    Returns:
    - out: pandas.DataFrame
        DataFrame containing:
        - 'flat vols': Given flat volatilities.
        - 'cap prices': Computed cap prices from Black's formula.
        - 'fwd vols': Implied forward volatilities for caplets.
    - caplets: pandas.DataFrame (only if returnCaplets=True)
        DataFrame containing caplet prices at different maturities.

    Notes:
    - This function extracts forward volatilities by ensuring that the cap prices (sum of caplet prices) match
      those implied by the flat volatilities.
    - It uses an iterative approach: 
      1. Computes cap prices using `cap_vol_to_price`.
      2. Extracts the caplet prices.
      3. Solves for the forward volatility of each caplet using `fsolve`.

    """

    # Warning for incorrect frequency setting
    if freq != 4:
        display('Warning: freq parameter controls time grid and cap timing.')

    dt = 1 / freq  # Time interval per caplet (e.g., quarterly = 0.25)

    # Initialize output DataFrames
    out = pd.DataFrame(dtype=float, index=flatvols.index, columns=['fwd vols', 'cap prices'])
    caplets = pd.DataFrame(dtype=float, index=flatvols.index, columns=strikes.values)

    # Identify the first cap maturity that contributes to caplet pricing
    first_cap = flatvols.index.get_loc(2 * dt)

    # Iterate through each time step to extract forward volatilities
    for step, t in enumerate(flatvols.index):
        if step < first_cap:
            # Before the first cap maturity, forward vols and cap prices are undefined
            out.loc[t, 'cap prices'] = np.nan
            out.loc[t, 'fwd vols'] = np.nan
            tprev = t
        else:
            # Compute the cap price using Black's formula
            out.loc[t, 'cap prices'] = fv.cap_vol_to_price(flatvols.loc[t], strikes.loc[t], 
                                                        fwds.loc[:t], discounts.loc[:t], 
                                                        dt=dt, notional=notional)

            if step == first_cap:
                # First cap maturity: Forward vol is the same as the flat vol
                out.loc[t, 'fwd vols'] = flatvols.loc[t]
                caplets.loc[t, strikes.loc[t]] = out.loc[t, 'cap prices']
                tprev = t
            else:
                strikeT = strikes.loc[t]

                # Compute caplet prices for previous time steps
                for j in flatvols.index[first_cap:step]:
                    caplets.loc[j, strikeT] = price_caplet(j - dt, out.loc[j, 'fwd vols'], 
                                                           strikeT, fwds.loc[j], discounts.loc[j], 
                                                           freq=freq, notional=notional)

                # Compute the current caplet price as the difference between cap prices
                caplets.loc[t, strikeT] = out.loc[t, 'cap prices'] - caplets.loc[:tprev, strikeT].sum()

                # Define a function to solve for the forward volatility
                wrapper = lambda vol: caplets.loc[t, strikeT] - price_caplet(tprev, vol, 
                                                                             strikeT, fwds.loc[t], 
                                                                             discounts.loc[t], 
                                                                             freq=freq, notional=notional)

                # Solve for forward volatility using fsolve
                out.loc[t, 'fwd vols'] = fsolve(wrapper, out.loc[tprev, 'fwd vols'])[0]            

                # Update previous time step
                tprev = t            

    # Insert flat volatilities into the output DataFrame
    out.insert(0, 'flat vols', flatvols)

    # Return results
    if returnCaplets:
        return out, caplets
    else:
        return out

# price a swaption
#  1. GET THAT WEIRD ASS DISCOUNT FACTOR
def calc_swaption_discount_factor(n, cap_curves, expiration, tenor):
    """
    Calculate the swaption discount factor.

    Parameters:
    n (int): Frequency of payments per year.
    cap_curves (pd.DataFrame): DataFrame with discount factors, with tenors as index.
    expiration (float): Maturity of the swaption in years.
    tenor (float): Maturity of the swap in years (relative to today).

    Returns:
    float: The swaption discount factor.
    """
    """
    n = frequency
    cap_curves = dataframe with discount factors, with tenors as index
    t = current time
    T_fwd = maturity of swaption
    T = maturity of swap (relative to today)
    """
    T_fwd = expiration
    T = expiration + tenor
    
    swaption_discount_factor = cap_curves.loc[T_fwd+(1/n):T, 'discounts'].sum()

    return swaption_discount_factor

# 2. CALC FORWARD SWAP RATE TO PLUG INTO BLACKS
def calc_forward_swap_rate(n, cap_curves, expiration, tenor):
    """
    Calculate the forward swap rate to plug into Black's formula.

    Parameters:
    n (int): Frequency of payments per year.
    cap_curves (pd.DataFrame): DataFrame with discount factors, with tenors as index.
    expiration (float): Maturity of the swaption in years.
    tenor (float): Maturity of the swap in years (relative to today).

    Returns:
    float: The forward swap rate.
    """
    """
    n = frequency
    cap_curves = dataframe with discount factors, with tenors as index
    t = current time
    T_fwd = maturity of swaption
    T = maturity of swap (relative to today)
    """
    T_fwd = expiration
    T = expiration + tenor
    
    Z_1 = cap_curves.loc[T_fwd, 'discounts']
    Z_2 = cap_curves.loc[T, 'discounts']
    swaption_discount_factor = calc_swaption_discount_factor(n, cap_curves, expiration, tenor)
    forward_swap_rate = n * (Z_1 - Z_2) / swaption_discount_factor

    return forward_swap_rate

# 3. CALC SWAPTION PRICE ACROSS STRIKES
def calc_swaption_price(T, implied_vol, forward_swap_rate, swaption_discount_factor, notional, frequency, strike_diff):
    """
    Calculate the price of a swaption using Black's formula.

    Parameters:
    T (float): Time to maturity of the swaption in years.
    implied_vol (float): Implied volatility of the swaption as a percentage (e.g., 20 for 20%).
    forward_swap_rate (float): The forward swap rate.
    swaption_discount_factor (float): The discount factor for the swaption.
    notional (float): The notional amount of the swaption.
    frequency (int): Frequency of payments per year (e.g., 4 for quarterly).
    strike_diff (float): Difference between the strike rate and the forward swap rate as a percentage (e.g., 1 for 1%).

    Returns:
    float: The price of the swaption.
    """
    return notional/frequency * blacks_formula(T,
                                                      implied_vol/100,
                                                      forward_swap_rate + (strike_diff/100/100),
                                                      forward_swap_rate,
                                                      swaption_discount_factor,
                                                      isCall=True)

def calc_swaptions_across_strikes(T,implied_vols,forward_swap_rate,swaption_discount_factor, notional, frequency):
    """
    Calculate swaption prices across different strikes.

    Parameters:
    T (float): Time to maturity in years.
    implied_vols (dict): Dictionary containing strike differences as keys and implied volatilities as values.
    forward_swap_rate (float): The forward swap rate.
    swaption_discount_factor (float): The swaption discount factor.
    notional (float): Notional amount.
    frequency (int): Frequency of payments per year.

    Returns:
    dict: A dictionary where the keys are the strike differences and the values are dictionaries containing
          the strike, implied volatility, and swaption price.
    """
    """
    implied vols is dict which contains strike diff as keys and implied vol as values
    """
    swaption_prices = {}
    for strike_diff, implied_vol in implied_vols.items():
        swaption_prices[strike_diff] = {'strike': forward_swap_rate + (strike_diff/100/100),
                                        'implied vol': implied_vol/100,
                                        'price': notional/frequency * blacks_formula(T,
                                                      implied_vol/100,
                                                      forward_swap_rate + (strike_diff/100/100),
                                                      forward_swap_rate,
                                                      swaption_discount_factor,
                                                      isCall=True)}
    return swaption_prices

# SABR
#  usage is in homework 2
#  refer to google doc of notes


# STIR Futures

# Treasury Futures

# Binomial Trees





def construct_bond_cftree(T, compound, cpn, cpn_freq=2, face=100, drop_final_period=True):
    """
    Constructs a cash flow tree for a bond.

    Parameters:
    T (float): The maturity time of the bond.
    compound (int): The compounding frequency per year.
    cpn (float): The annual coupon rate of the bond.
    cpn_freq (int, optional): The frequency of coupon payments per year. Default is 2.
    face (float, optional): The face value of the bond. Default is 100.
    drop_final_period (bool, optional): If True, the final period cash flow is dropped. Default is True.

    Returns:
    pd.DataFrame: A DataFrame representing the cash flow tree of the bond.
    """
    
    def construct_rate_tree(dt, T):
        """
        Creates an empty tree with a time grid.

        Parameters:
        dt (float): The time step size.
        T (float): The maturity time of the bond.

        Returns:
        pd.DataFrame: A DataFrame representing the empty rate tree.
        """
        timegrid = pd.Series((np.arange(0, round(T/dt) + 1) * dt).round(6), name='time', index=pd.Index(range(round(T/dt) + 1), name='state'))
        tree = pd.DataFrame(dtype=float, columns=timegrid, index=timegrid.index)
        return tree
    
    step = int(compound / cpn_freq)

    cftree = construct_rate_tree(1 / compound, T)
    cftree.iloc[:, :] = 0
    cftree.iloc[:, -1:0:-step] = (cpn / cpn_freq) * face
    
    if drop_final_period:
        # Final cash flow is accounted for in payoff function
        # Drop final period cash flow from cash flow tree
        cftree = cftree.iloc[:-1, :-1]
    else:
        cftree.iloc[:, -1] += face
        
    return cftree

def construct_accint(timenodes, freq, cpn, cpn_freq=2, face=100):
    """
    Constructs the accrued interest for a bond over given time nodes.

    Parameters:
    timenodes (array-like): The time nodes at which to calculate accrued interest.
    freq (int): The frequency of compounding per year.
    cpn (float): The annual coupon rate of the bond.
    cpn_freq (int, optional): The frequency of coupon payments per year. Default is 2.
    face (float, optional): The face value of the bond. Default is 100.

    Returns:
    pd.Series: A Series representing the accrued interest at each time node.
    """
    
    mod = freq / cpn_freq
    cpn_pmnt = face * cpn / cpn_freq

    temp = np.arange(len(timenodes)) % mod
    # Shift to ensure end is considered coupon (not necessarily start)
    temp = (temp - temp[-1] - 1) % mod
    temp = cpn_pmnt * temp.astype(float) / mod

    accint = pd.Series(temp, index=timenodes)

    return accint

import pandas as pd
import numpy as np

def compute_bond_price_tree(ratetree, cftree, face_value=100, coupon_rate=0.0441, coupon_freq=2):
    """
    Computes a tree of bond prices using backward induction, accounting for the final cash flow at maturity.

    Parameters:
    -----------
    ratetree : pandas.DataFrame
        A binomial tree of short-term interest rates (annualized).
        
    cftree : pandas.DataFrame
        A matrix of bond cash flows at each time step (excluding final maturity payment).

    face_value : float, default=100
        The face value of the bond.

    coupon_rate : float, default=0.0441
        The bond's annual coupon rate (e.g., 0.0441 for 4.41%).

    coupon_freq : int, default=2
        The number of coupon payments per year (e.g., 2 for semiannual payments).

    Returns:
    --------
    valuetree : pandas.DataFrame
        A tree containing bond prices at each time step.
    """
    # Ensure the input trees have the same shape
    assert ratetree.shape == cftree.shape, "Rate tree and cash flow tree must have the same shape."

    # Time steps and interval
    dt = ratetree.columns[1] - ratetree.columns[0]
    final_timestep = cftree.columns[-1]  # Last available column (t = 4.75)

    # Initialize the value tree with the same structure
    valuetree = pd.DataFrame(dtype=float, index=ratetree.index, columns=ratetree.columns)

    # Compute the final bond cash flow at maturity (t = 5)
    final_cashflow = face_value + (coupon_rate / coupon_freq) * face_value  # 102.25
    discount_factor_final = np.exp(-ratetree.iloc[:, -1] * dt)  # Discount using t=4.75 rate
    valuetree.iloc[:, -1] = final_cashflow * discount_factor_final  # Discounted at t=4.75 rate

    # Backward induction to compute bond prices
    for t in reversed(range(len(valuetree.columns) - 1)):
        time_col = valuetree.columns[t]
        next_time_col = valuetree.columns[t + 1]

        for state in range(len(valuetree.index) - 1):
            p = 0.5  # Risk-neutral probability

            # Expected future value (up and down states)
            expected_value = (
                p * valuetree.loc[state, next_time_col] + 
                (1 - p) * valuetree.loc[state + 1, next_time_col]
            )

            # Discount at the current rate
            discount_factor = np.exp(-ratetree.loc[state, time_col] * dt)

            # Compute bond price as discounted expectation + coupon payment
            valuetree.loc[state, time_col] = discount_factor * expected_value + cftree.loc[state, time_col]

    return valuetree



# def bond_price(face_value, coupon_rate, ytm, years_to_maturity):
#     coupon_payment = (coupon_rate / 2) * face_value
#     periods = years_to_maturity * 2
#     price = sum([coupon_payment / (1 + ytm / 2) ** t for t in range(1, periods + 1)])
#     price += face_value / (1 + ytm / 2) ** periods
#     return price


def bintree_pricing(payoff=None, ratetree=None, undertree=None, cftree=None, dt=None, pstars=None, timing=None, cfdelay=False, style='european', Tamerican=0, compounding=None):
    """
    Prices a derivative using a binomial tree model.

    Parameters:
    -----------
    payoff : function, optional
        A function that calculates the payoff of the derivative given the underlying rate.
        
    ratetree : pandas.DataFrame, optional
        A binomial tree of short-term interest rates (annualized).
        
    undertree : pandas.DataFrame, optional
        A binomial tree of the underlying asset prices. Defaults to ratetree if not provided.
        
    cftree : pandas.DataFrame, optional
        A tree containing cash flows at each time step. Defaults to a zero cash flow tree if not provided.
        
    dt : float, optional
        The time step size. If not provided, it is calculated from the columns of undertree.
        
    pstars : pandas.Series, optional
        The risk-neutral probabilities at each time step. Defaults to 0.5 if not provided.
        
    timing : str, optional
        The timing of cash flows. If 'deferred', cash flows are delayed by dt.
        
    cfdelay : bool, optional
        If True, cash flows are delayed by dt. Defaults to False.
        
    style : str, optional
        The option style. Can be 'european' or 'american'. Defaults to 'european'.
        
    Tamerican : int, optional
        The time step at which American option exercise is allowed. Defaults to 0.
        
    compounding : int, optional
        The compounding frequency. If provided, ratetree is converted to continuous rates.

    Returns:
    --------
    valuetree : pandas.DataFrame
        A tree containing the derivative prices at each time step.
    """
    
    if payoff is None:
        payoff = lambda r: 0
    
    if undertree is None:
        undertree = ratetree
        
    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)
        
    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if dt is None:
        dt = undertree.columns.to_series().diff().mean()
        dt = undertree.columns[1] - undertree.columns[0]
    
    if timing == 'deferred':
        cfdelay = True
    
    if dt < .25 and cfdelay:
        display('Warning: cfdelay setting only delays by dt.')
        
    if compounding is not None:
        ratetree_cont = compounding * np.log(1 + ratetree / compounding)
    else:
        ratetree_cont = ratetree
    
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back == 0:                           
            valuetree[t] = payoff(undertree[t])
            if cfdelay:
                valuetree[t] *= np.exp(-ratetree_cont[t] * dt)
        else:
            for state in valuetree[t].index[:-1]:
                val_avg = pstars[t] * valuetree.iloc[state, -steps_back] + (1 - pstars[t]) * valuetree.iloc[state + 1, -steps_back]
                
                if cfdelay:
                    cf = cftree.loc[state, t]
                else:                    
                    cf = cftree.iloc[state, -steps_back]
                
                valuetree.loc[state, t] = np.exp(-ratetree_cont.loc[state, t] * dt) * (val_avg + cf)

            if style == 'american':
                if t >= Tamerican:
                    valuetree.loc[:, t] = np.maximum(valuetree.loc[:, t], payoff(undertree.loc[:, t]))
        
    return valuetree





def payoff_bond(r,dt,facevalue=100):
    price = np.exp(-r * dt) * facevalue
    return price 


def payoff_call(r, dt, facevalue=100, k=100):
    price = np.maximum(np.exp(-r * dt) * facevalue - k, 0)
    return price
