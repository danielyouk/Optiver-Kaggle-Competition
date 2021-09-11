library(data.table)
library(tidyverse)
library(readr)
library(plotly)
library(ggplot2)
library(tictoc)
library(reticulate)
library(arrow)
library(tictoc)
library(readr)
library(tidymodels)
library(lightgbm)
list.of.packages <- c(
  "foreach",
  "doParallel",
  "ranger",
  "palmerpenguins",
  "tidyverse",
  "kableExtra"
)

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages) > 0){
  install.packages(new.packages, dep=TRUE)
}

for(package.i in list.of.packages){
  suppressPackageStartupMessages(
    library(
      package.i, 
      character.only = TRUE
    )
  )
}

list.of.packages <- c(
  "janitor",
  "dplyr",
  "ggplot2",
  "rsample",
  "recipes",
  "parsnip",
  "tune",
  "dials",
  "workflows",
  "yardstick",
  "treesnip"
)

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages) > 0){
  install.packages(new.packages, dep=TRUE)
}

for(package.i in list.of.packages){
  suppressPackageStartupMessages(
    library(
      package.i, 
      character.only = TRUE
    )
  )
}
# file_name <- list.files('./input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
# book_example <- read_parquet(paste0("./input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0/",file_name)) %>% as.data.table()
# file_name <- list.files('./input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0')
# trade_example <-  read_parquet(paste0("./input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0/",file_name)) %>% as.data.table()

root_dir <- './input/optiver-realized-volatility-prediction/book_train.parquet'
stock_id_list <- list.files(root_dir)
# 1. data processing ----
preprocessor <- function (stock_id){
  #stock_id <- "stock_id=0"
  file_name <- list.files(paste0("./input/optiver-realized-volatility-prediction/book_train.parquet/",stock_id))
  book_example <- read_parquet(paste0("./input/optiver-realized-volatility-prediction/book_train.parquet/",stock_id,"/",file_name)) %>% as.data.table()
  file_name <- list.files(paste0("./input/optiver-realized-volatility-prediction/trade_train.parquet/",stock_id))
  trade_example <- read_parquet(paste0("./input/optiver-realized-volatility-prediction/trade_train.parquet/",stock_id,"/",file_name)) %>% as.data.table()
  stock_id <- str_split(stock_id,"=")[[1]][2] 
  
  book_example[,row_id:=paste0(stock_id,"-",time_id)]
  book_example[,wap1:=(bid_price1*ask_size1 + ask_price1 * bid_size1)/(bid_size1 + ask_size1)]
  book_example[,wap2:=(bid_price2*ask_size2 + ask_price2 * bid_size2)/(bid_size2 + ask_size2)]
  book_example[,wap3:=(bid_price1*bid_size1 + ask_price1 * ask_size1)/(bid_size1 + ask_size1)]
  book_example[,wap4:=(bid_price2*bid_size2 + ask_price2 * ask_size2)/(bid_size2 + ask_size2)]
  
  book_example[,wap1_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap1"), by=.(row_id)]
  book_example[,wap2_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap2"), by=.(row_id)]
  book_example[,wap3_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap3"), by=.(row_id)]
  book_example[,wap4_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("wap4"), by=.(row_id)]
  
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
                       wap1_std = sd(wap1),
                       wap2_sum = sum(wap2), 
                       wap2_std = sd(wap2),
                       wap3_sum = sum(wap3), 
                       wap3_std = sd(wap3),
                       wap4_sum = sum(wap4), 
                       wap4_std = sd(wap4),
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
    by=.(row_id)]
  
  
  book_example_summary_timegroup <- 
    book_example[,list(log_return1_realized_volatility = sqrt(sum(log_return1**2, na.rm = T)),
                       log_return2_realized_volatility = sqrt(sum(log_return2**2, na.rm = T)),
                       log_return3_realized_volatility = sqrt(sum(log_return3**2, na.rm = T)),
                       log_return4_realized_volatility = sqrt(sum(log_return4**2, na.rm = T))
    ), by=.(row_id, time_group)]
  
  
  book_example_summary_timegroup_wide <- 
    book_example_summary_timegroup %>% 
    pivot_wider(names_from = time_group, 
                values_from = c(log_return1_realized_volatility,log_return2_realized_volatility,log_return3_realized_volatility,log_return4_realized_volatility))
  
  

  trade_example[,row_id:=paste0(stock_id,"-",time_id)]
  trade_example[,price_lag := shift(.SD, 1, NA, "lag"), .SDcols=c("price"), by=.(row_id)]
  trade_example[,log_return:=log(price/price_lag)]
  trade_example[,amount:= price*size]
  trade_example[,time_group := ifelse(seconds_in_bucket>=500,500,
                                      ifelse(seconds_in_bucket>=400,400,
                                             ifelse(seconds_in_bucket>=300,300,
                                                    ifelse(seconds_in_bucket>=200,200,100))))]
  
  trade_example[,price_mean:=mean(price), by=.(row_id)]
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
                        trade_order_count_amax = max(order_count),
                        trade_size_sum = sum(size),
                        trade_size_amax = max(size),
                        trade_size_amin = min(size),
                        trade_tendency = sum((price_diff/price)*100 *size, na.rm = T),
                        trade_abs_diff = median(abs(price - price_mean)),
                        trade_energy = mean(price**2),
                        trade_iqr_p = quantile(price, 0.75) - quantile(price, 0.25),
                        trade_abs_diff_v = median(abs(size - mean(size))),
                        trade_energy_v = mean(size**2),
                        trade_iqr_p_v = quantile(size, 0.75) - quantile(size, 0.25)
    ),
    by=.(row_id)]
  
  
  trade_example_summary_timegroup <- 
    trade_example[,list(trade_log_return_realized_volatility = sqrt(sum(log_return**2, na.rm = T)),
                        trade_seconds_in_bucket_count_unique = .N,
                        trade_order_count_sum = sum(order_count),
                        trade_size_sum = sum(size)
    ), 
    by=.(row_id, time_group)]
  
  selcols <- setdiff(names(trade_example_summary_timegroup), c("row_id","time_group"))
  trade_example_summary_timegroup_wide <- 
    trade_example_summary_timegroup %>% 
    pivot_wider(names_from = time_group, 
                values_from = all_of(selcols)) %>% 
    as.data.table()
  
  trade_example_summary_pricegroup <- 
    trade_example[,list(f_ = sum(price)),by=.(row_id, price_group)] %>% na.omit()
  
  trade_example_summary_pricegroup_wide <- 
    trade_example_summary_pricegroup %>% 
    pivot_wider(names_from = price_group, values_from = c(f_)) %>% 
    as.data.table() %>% 
    setnames(c("min","max"),c("trade_f_min","trade_f_max"))
  
  trade_example_summary_pricegroup2 <- 
    trade_example[,list(df_ = .N), by=.(row_id, price_group2)] %>% na.omit()
  
  trade_example_summary_pricegroup2_wide <- 
    trade_example_summary_pricegroup2 %>% 
    pivot_wider(names_from = price_group2, values_from = c(df_)) %>% 
    as.data.table() %>% 
    setnames(c("min","max"),c("trade_df_min","trade_df_max"))
  
  train <- book_example_summary %>% 
    merge(book_example_summary_timegroup_wide, by=c("row_id"), all.x = T) %>% 
    merge(trade_example_summary, by=c("row_id"), all.x = T) %>% 
    merge(trade_example_summary_pricegroup_wide, by=c("row_id"), all.x = T) %>% 
    merge(trade_example_summary_pricegroup2_wide, by=c("row_id"), all.x = T) %>% 
    merge(trade_example_summary_timegroup_wide, by=c("row_id"), all.x = T)
  
  return(train)
}

# tic()
# train <- data.table()
# i <-0
# for (stock_id_input in stock_id_list[1:10]){
#   i <- i +1; t1 <- Sys.time()
#   train_each <- preprocessor(stock_id = stock_id_input)
#   train <- rbind(train, train_each)
#   t2 <- Sys.time()
#   timediff <- difftime(t2,t1, units = "secs")
#   print(paste0(i, " duration ",timediff, " sec"))
#   
# }
# toc()

# 2. dopar processing ----
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
print(my.cluster)

tic()
doParallel::registerDoParallel(cl = my.cluster)

train_ <- foreach(
  stock_id_input = stock_id_list,
  .combine = "rbind",
  .packages = c("tidyverse","data.table","arrow")
) %dopar% {
  preprocessor(stock_id = stock_id_input)
}

parallel::stopCluster(cl = my.cluster)
toc()
# cols_python_object <- names(py$train_)
# cols_r_object <-names(train)
# 
# setdiff(cols_python_object, cols_r_object)
# setdiff(cols_r_object, cols_python_object)

train <- read_csv("input/optiver-realized-volatility-prediction/train.csv") %>% as.data.table()
train[,row_id:=paste0(stock_id,"-",time_id)]
train <- merge(train_, train[,-c("stock_id","time_id")], by=c("row_id"), all.x = T)

train[,size_tau:=sqrt(1/trade_seconds_in_bucket_count_unique)]
train[,size_tau_500:=sqrt(1/trade_seconds_in_bucket_count_unique_500)]
train[,size_tau_400:=sqrt(1/trade_seconds_in_bucket_count_unique_400)]
train[,size_tau_300:=sqrt(1/trade_seconds_in_bucket_count_unique_300)]
train[,size_tau_200:=sqrt(1/trade_seconds_in_bucket_count_unique_200)]
train[,size_tau_100:=sqrt(1/trade_seconds_in_bucket_count_unique_100)]

rmspe_vec <- function(truth, estimate, na_rm = TRUE, ...) {
  
  rmspe_impl <- function(truth, estimate) {
    sqrt(mean(((truth - estimate)/truth) ^ 2))
  }
  
  metric_vec_template(
    metric_impl = rmspe_impl,
    truth = truth, 
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
  
}

rmspe_vec(truth = c(NA, .5, .4), estimate = c(1,.6,.5))

library(rlang)

rmspe <- function(data, ...) {
  UseMethod("rmspe")
}

rmspe <- new_numeric_metric(rmspe, direction = "minimize")

rmspe.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  
  metric_summarizer(
    metric_nm = "rmspe",
    metric_fn = rmspe_vec,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    na_rm = na_rm,
    ...
  )
  
}



Optiver_data_split <- rsample::initial_split(
  train,
  prop = 0.8,
  strata = target
)

preprocessing_recipe <-
  recipes::recipe(target ~ ., data = training(Optiver_data_split)) %>%
  recipes::step_rm(row_id) %>% 
  recipes::step_normalize(all_numeric()) %>%
  prep()

Optiver_data_cv_folds <-
  recipes::bake(
    preprocessing_recipe,
    new_data = training(Optiver_data_split)
  ) %>%
  rsample::vfold_cv(v = 5)



lightgbm_model<-
  parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
  ) %>%
  set_engine("lightgbm")

lightgbm_params <-
  dials::parameters(
    # The parameters have sane defaults, but if you have some knowledge 
    # of the process you can set upper and lower limits to these parameters.
    min_n(), # 2nd important
    tree_depth() # 3rd most important
  )

set.seed(1234)
lgbm_grid <-
  dials::grid_max_entropy(
    lightgbm_params,
    size = 30 # set this to a higher number to get better results
    # I don't want to run this all night, so I set it to 30
  )
head(lgbm_grid)

lgbm_wf <-
  workflows::workflow() %>%
  add_model(lightgbm_model
  ) %>%
  add_formula(target ~ .)

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
tic()
doParallel::registerDoParallel(cl = my.cluster)
unregister_dopar()
lgbm_tuned <- tune::tune_grid(
  object = lgbm_wf,
  resamples = Optiver_data_cv_folds,
  grid = lgbm_grid,
  metrics = yardstick::metric_set(rmse, mae),
  control = tune::control_grid(verbose = FALSE) # set this to TRUE to see
  # in what step of the process you are. But that doesn't look that well in
  # a blog.
)

parallel::stopCluster(cl = my.cluster)
toc()