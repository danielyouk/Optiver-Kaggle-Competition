library(data.table)
library(tidyverse)
library(readr)
library(plotly)
library(ggplot2)
library(tictoc)
library(reticulate)
library(arrow)


file_name <- list.files('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
book_example <- read_parquet(paste0("../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0/",file_name)) %>% as.data.table()
file_name <- list.files('../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0')
trade_example <-  read_parquet(paste0("../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0/",file_name)) %>% as.data.table()

# 1. book data processing ----
book_example[,wap1:=(bid_price1*ask_size1 + ask_price1 * bid_size1)/(bid_size1 + ask_size1)]
book_example[,wap2:=(bid_price2*ask_size2 + ask_price2 * bid_size2)/(bid_size2 + ask_size2)]
book_example[,wap3:=(bid_price1*bid_size1 + ask_price1 * ask_size1)/(bid_size1 + ask_size1)]
book_example[,wap4:=(bid_price2*bid_size2 + ask_price2 * ask_size2)/(bid_size2 + ask_size2)]

book_example[,wap1_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap1"), by=.(time_id)]
book_example[,wap2_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap2"), by=.(time_id)]
book_example[,wap3_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap3"), by=.(time_id)]
book_example[,wap4_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap4"), by=.(time_id)]

book_example[,log_return1:= log(wap1/wap1_lag)]
book_example[,log_return2:= log(wap2/wap2_lag)]
book_example[,log_return3:= log(wap3/wap3_lag)]
book_example[,log_return4:= log(wap4/wap4_lag)]

book_example[,wap_balance := abs(wap1 - wap2)]

book_example[,price_spread := (ask_price1 - bid_price1)/((ask_price1 + bid_price1)/2)]
book_example[,price_spread2 := (ask_price2 - bid_price2)/((ask_price2 + bid_price2)/2)]

book_example[,bid_spread := bid_price1 - bid_price2]
book_example[,ask_spread := ask_price1 - ask_price2]

book_example[,total_volume := ask_size1 + ask_size2 + bid_size1 + bid_size2]
book_example[,volume_imbalance := abs((ask_size1 + ask_size2)-(bid_size1 + bid_size2))]

book_example[,bid_ask_spread := abs(bid_spread - ask_spread)]

book_example[,time_group := ifelse(seconds_in_bucket>=500,500,
                                   ifelse(seconds_in_bucket>=400,400,
                                          ifelse(seconds_in_bucket>=300,300,
                                                 ifelse(seconds_in_bucket>=200,200,100))))]

book_example_summary <- 
  book_example[,list(wap1_sum = sum(wap1), 
                   wap1_sd = sd(wap1),
                   wap2_sum = sum(wap2), 
                   wap2_sd = sd(wap2),
                   wap3_sum = sum(wap3), 
                   wap3_sd = sd(wap3),
                   wap4_sum = sum(wap4), 
                   wap4_sd = sd(wap4),
                   log_return1_realized_volatility = sqrt(sum(log_return1**2, na.rm = T)),
                   log_return2_realized_volatility = sqrt(sum(log_return2**2, na.rm = T)),
                   log_return3_realized_volatility = sqrt(sum(log_return3**2, na.rm = T)),
                   log_return4_realized_volatility = sqrt(sum(log_return4**2, na.rm = T)),
                   wap_balance_sum = sum(wap_balance),
                   wap_balance_amax = max(wap_balance),
                   price_spread_sum = sum(price_spread),
                   price_spread_amax = max(price_spread),
                   price_spread2_sum = sum(price_spread2),
                   price_spread2_amax = max(price_spread2),
                   bid_spread_sum = sum(bid_spread),
                   bid_spread_amax = max(bid_spread),
                   ask_spread_sum = sum(ask_spread),
                   ask_spread_amax = max(ask_spread),
                   total_volume_sum = sum(total_volume),
                   total_volume_amax = max(total_volume),
                   volume_imbalance_sum = sum(volume_imbalance),
                   volume_imbalance_amax = max(volume_imbalance),
                   bid_ask_spread_sum = sum(bid_ask_spread),
                   bid_ask_spread_amax = max(bid_ask_spread)
                   ), 
             by=.(time_id)]


book_example_summary_timegroup <- 
  book_example[,list(log_return1_realized_volatility = sqrt(sum(log_return1**2, na.rm = T)),
                     log_return2_realized_volatility = sqrt(sum(log_return2**2, na.rm = T)),
                     log_return3_realized_volatility = sqrt(sum(log_return3**2, na.rm = T)),
                     log_return4_realized_volatility = sqrt(sum(log_return4**2, na.rm = T))
                     ), by=.(time_id, time_group)]


book_example_summary_timegroup_wide <- 
  book_example_summary_timegroup %>% 
  pivot_wider(names_from = time_group, 
              values_from = c(log_return1_realized_volatility,log_return2_realized_volatility,log_return3_realized_volatility,log_return4_realized_volatility))


# 2. trade data processing ----
trade_example[,price_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("price"), by=.(time_id)]
trade_example[,log_return:=log(price/price_lag)]
trade_example[,amount:= price*size]
trade_example[,time_group := ifelse(seconds_in_bucket>=500,500,
                                   ifelse(seconds_in_bucket>=400,400,
                                          ifelse(seconds_in_bucket>=300,300,
                                                 ifelse(seconds_in_bucket>=200,200,100))))]

trade_example[,price_mean:=mean(price), by=.(time_id)]
trade_example[price > price_mean, price_group:="max"]
trade_example[price < price_mean, price_group:="min"]
trade_example[,price_diff:=price-price_lag]
trade_example[price_diff > 0,price_group2:="max"]
trade_example[price_diff < 0,price_group2:="min"]

trade_example_summary <-
  trade_example[,list(trade_log_return_realized_volatility = sqrt(sum(log_return**2, na.rm = T)),
                      trade_seconds_in_bucket_count_unique = .N,
                      trade_amount_sum = sum(amount),
                      trade_amount_amax = max(amount),
                      trade_amount_amin = min(amount),
                      trade_order_count_sum = sum(order_count),
                      trade_size_sum = sum(size),
                      trade_size_amax = max(size),
                      trade_size_amin = min(size),
                      trade_tendency = sum((price_diff/price)*100 *size, na.rm = T),
                      trade_abs_diff = median(abs(price - price_mean)),
                      trade_energy = mean(price**2)
                      ),
                by=.(time_id)]


trade_example_summary_timegroup <- 
  trade_example[,list(trade_log_return_realized_volatility = sqrt(sum(log_return**2, na.rm = T)),
                      trade_seconds_in_bucket_count_unique = .N,
                      trade_order_count_sum = sum(order_count),
                      trade_size_sum = sum(size)
                      ), 
                by=.(time_id, time_group)]

selcols <- setdiff(names(trade_example_summary_timegroup), c("time_id","time_group"))
trade_example_summary_timegroup_wide <- 
  trade_example_summary_timegroup %>% 
  pivot_wider(names_from = time_group, 
              values_from = all_of(selcols)) %>% 
  as.data.table()

trade_example_summary_pricegroup <- 
  trade_example[,list(f_ = sum(price)),by=.(time_id, price_group)] %>% na.omit()
trade_example_summary_pricegroup2 <- 
  trade_example[,list(df_ = .N), by=.(time_id, price_group2)] %>% na.omit()

trade_example_summary_pricegroup_wide <- 
  trade_example_summary_pricegroup %>% 
  pivot_wider(names_from = price_group, values_from = c(f_)) %>% 
  as.data.table() %>% 
  setnames(c("min","max"),c("trade_f_min","trade_f_max"))

trade_example_summary_pricegroup2_wide <- 
  trade_example_summary_pricegroup2 %>% 
  pivot_wider(names_from = price_group2, values_from = c(df_)) %>% 
  as.data.table() %>% 
  setnames(c("min","max"),c("trade_df_min","trade_df_max"))


