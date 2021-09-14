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
library(tidymodasdfasels)
library(lightgbm)
library(xgboost)
require(xgboost)
require(Matrix)
require(data.table)
library(Ckmeans.1d.dp)
if (!require('vcd')) install.packages('vcd')

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
  #stock_id = "stock_id=0"
  file_name <- list.files(paste0("./input/optiver-realized-volatility-prediction/book_train.parquet/",stock_id))
  book_example <- read_parquet(paste0("./input/optiver-realized-volatility-prediction/book_train.parquet/",stock_id,"/",file_name)) %>% as.data.table()
  file_name <- list.files(paste0("./input/optiver-realized-volatility-prediction/trade_train.parquet/",stock_id))
  trade_example <- read_parquet(paste0("./input/optiver-realized-volatility-prediction/trade_train.parquet/",stock_id,"/",file_name)) %>% as.data.table()
  
  book_trade_merge <- merge(book_example,trade_example,by=c("time_id","seconds_in_bucket"), all = TRUE)
  stock_id_ <- str_split(stock_id,"=")[[1]][2]
  
  book_trade_merge[, WAP1:=(bid_price1*ask_size1 + ask_price1 * bid_size1)/(bid_size1 + ask_size1)]
  book_trade_merge[, WAP2:=(bid_price2*ask_size2 + ask_price2 * bid_size2)/(bid_size2 + ask_size2)]
  book_trade_merge[, Bid1_Ask1_Spread:= ask_price1 - bid_price1]
  book_trade_merge[, Bid1_Ask1_Margin:= (Bid1_Ask1_Spread/ask_price1)*100]
  book_trade_merge[, Bid2_Ask2_Spread:= ask_price2 - bid_price2]
  book_trade_merge[, Bid2_Ask2_Margin:= (Bid2_Ask2_Spread/ask_price2)*100]
  book_trade_merge[, WAP1_price_Spread:= price - WAP1]
  book_trade_merge[, WAP1_price_Margin:= (WAP1_price_Spread/price)*100]
  
  book_trade_merge[, count_BuySell:=ifelse(is.na(order_count),0,1)]
  book_trade_merge[, transaction_amount:=price * size]
  book_trade_merge[, WAP1_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("WAP1"), by=.(time_id)]
  book_trade_merge[, WAP2_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("WAP2"), by=.(time_id)]
  book_trade_merge[, Bid1_Ask1_Spread_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("Bid1_Ask1_Spread"), by=.(time_id)]
  book_trade_merge[, Bid2_Ask2_Spread_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("Bid2_Ask2_Spread"), by=.(time_id)]  
  book_trade_merge[, Bid1_Ask1_Margin_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("Bid1_Ask1_Margin"), by=.(time_id)]
  book_trade_merge[, Bid2_Ask2_Margin_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("Bid2_Ask2_Margin"), by=.(time_id)]  
  book_trade_merge[, ask_price1_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("ask_price1"), by=.(time_id)]
  book_trade_merge[, ask_price2_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("ask_price2"), by=.(time_id)]
  book_trade_merge[, bid_price1_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("bid_price1"), by=.(time_id)]
  book_trade_merge[, bid_price2_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("bid_price2"), by=.(time_id)]  
  book_trade_merge[, bid_size1_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("bid_size1"), by=.(time_id)]
  book_trade_merge[, ask_size1_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("ask_size1"), by=.(time_id)]
  book_trade_merge[, bid_size2_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("bid_size2"), by=.(time_id)]
  book_trade_merge[, ask_size2_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("ask_size2"), by=.(time_id)]
  book_trade_merge[, price_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("price"), by=.(time_id)]
  book_trade_merge[, size_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("size"), by=.(time_id)]
  book_trade_merge[, order_count_lag:= shift(.SD, 1, NA, "lag"), .SDcols=c("order_count"), by=.(time_id)]
  
  time_ids <- book_trade_merge$time_id %>% unique()
  time_memory_whole <- data.table()
  for(sel_time_id in time_ids){
    #sel_time_id <- 11
    temp <- book_trade_merge[time_id == sel_time_id,]
    time_memory <- data.table()
    time_price_max_secs <- temp[which.max(temp$price)]$seconds_in_bucket %>% as.integer()
    time_price_min_secs <- temp[which.min(temp$price)]$seconds_in_bucket %>% as.integer()
    time_WAP1_max_secs <-temp[which.max(temp$WAP1)]$seconds_in_bucket %>% as.integer()
    time_WAP1_min_secs <-temp[which.min(temp$WAP1)]$seconds_in_bucket %>% as.integer()
    time_transaction_amount_max_secs <- temp[which.max(temp$size)]$seconds_in_bucket %>% as.integer()
    time_bid_size1_max_secs <- temp[which.max(temp$bid_size1)]$seconds_in_bucket %>% as.integer()
    time_ask_size1_max_secs <- temp[which.max(temp$ask_size1)]$seconds_in_bucket %>% as.integer()
    
    time_memory <- 
      time_memory[ ,c("time_id", "time_price_max", "time_price_min", "time_WAP1_max","time_WAP1_min", 
                      "time_transaction_amount_max", "time_bid_size1_max", "time_ask_size1_max") := 
                     list(sel_time_id, time_price_max_secs, time_price_min_secs, time_WAP1_max_secs, time_WAP1_min_secs, 
                          time_transaction_amount_max_secs, time_bid_size1_max_secs, time_ask_size1_max_secs)] %>% as.data.frame() %>% 
      pivot_longer(!time_id, names_to = "time",values_to = "seconds_in_bucket") %>% as.data.table() %>% 
      setorderv("seconds_in_bucket")
    
    time_memory_wide <- data.table()
    time_memory_wide[, c("time_id",
                         "time1","time2","time3","time4","time5","time6","time7",
                         "seconds_in_bucket1","seconds_in_bucket2","seconds_in_bucket3","seconds_in_bucket4","seconds_in_bucket5","seconds_in_bucket6","seconds_in_bucket7"):=
                       list(sel_time_id,
                            time_memory$time[1],time_memory$time[2], time_memory$time[3],time_memory$time[4],time_memory$time[5],time_memory$time[6],time_memory$time[7],
                            time_memory$seconds_in_bucket[1],time_memory$seconds_in_bucket[2], time_memory$seconds_in_bucket[3],time_memory$seconds_in_bucket[4],time_memory$seconds_in_bucket[5],
                            time_memory$seconds_in_bucket[6],time_memory$seconds_in_bucket[7])]
    
    time_memory_whole <- rbind(time_memory_whole,time_memory_wide)
  }
  
  book_trade_merge_summary <- 
    book_trade_merge[, list(
      
      ask_price1_start = first(na.omit(ask_price1)),
      ask_price1_max = max(ask_price1, na.rm = T),
      ask_price1_min = min(ask_price1, na.rm = T),
      ask_price1_last = last(na.omit(ask_price1)),
      ask_price1_log_return_volatility = sqrt(sum(log((ask_price1/ask_price1_lag))**2, na.rm = T)),     
  
      ask_price2_start = first(na.omit(ask_price2)),
      ask_price2_max = max(ask_price2, na.rm = T),
      ask_price2_min = min(ask_price2, na.rm = T),
      ask_price2_last = last(na.omit(ask_price2)),
      ask_price2_log_return_volatility = sqrt(sum(log((ask_price2/ask_price2_lag))**2, na.rm = T)),  
 
      bid_price1_start = first(na.omit(bid_price1)),
      bid_price1_max = max(bid_price1, na.rm = T),
      bid_price1_min = min(bid_price1, na.rm = T),
      bid_price1_last = last(na.omit(bid_price1)),
      bid_price1_log_return_volatility = sqrt(sum(log((bid_price1/bid_price1_lag))**2, na.rm = T)),     
      
      bid_price2_start = first(na.omit(bid_price2)),
      bid_price2_max = max(bid_price2, na.rm = T),
      bid_price2_min = min(bid_price2, na.rm = T),
      bid_price2_last = last(na.omit(bid_price2)),
      bid_price2_log_return_volatility = sqrt(sum(log((bid_price2/bid_price2_lag))**2, na.rm = T)), 
      
      WAP1_start = first(na.omit(WAP1)),
      WAP1_max = max(WAP1, na.rm = T),
      WAP1_min = min(WAP1, na.rm = T),
      WAP1_last = last(na.omit(WAP1)),
      WAP1_log_return_volatility = sqrt(sum(log((WAP1/WAP1_lag))**2, na.rm = T)),
      
      WAP2_start = first(na.omit(WAP2)),
      WAP2_max = max(WAP2, na.rm = T),
      WAP2_min = min(WAP2, na.rm = T),
      WAP2_last = last(na.omit(WAP2)),
      WAP2_log_return_volatility = sqrt(sum(log((WAP2/WAP2_lag))**2, na.rm = T)),
      
      bid_size1_start = first(na.omit(bid_size1)) %>% as.integer(),
      bid_size1_max = max(bid_size1, na.rm = T) %>% as.integer(),
      bid_size1_min = min(bid_size1, na.rm = T) %>% as.integer(),
      bid_size1_last = last(na.omit(bid_size1)) %>% as.integer(),     
      bid_size1_log_return_volatility = sqrt(sum(log((bid_size1/bid_size1_lag))**2, na.rm = T)),
      
      bid_size2_start = first(na.omit(bid_size2)) %>% as.integer(),
      bid_size2_max = max(bid_size2, na.rm = T) %>% as.integer(),
      bid_size2_min = min(bid_size2, na.rm = T) %>% as.integer(),
      bid_size2_last = last(na.omit(bid_size2)) %>% as.integer(),     
      bid_size2_log_return_volatility = sqrt(sum(log((bid_size2/bid_size2_lag))**2, na.rm = T)),
      
      ask_size1_start = first(na.omit(ask_size1)) %>% as.integer(),
      ask_size1_max = max(ask_size1, na.rm = T) %>% as.integer(),
      ask_size1_min = min(ask_size1, na.rm = T) %>% as.integer(),
      ask_size1_last = last(na.omit(ask_size1)) %>% as.integer(),  
      ask_size1_log_return_volatility = sqrt(sum(log((ask_size1/ask_size1_lag))**2, na.rm = T)),
      
      ask_size2_start = first(na.omit(ask_size2)) %>% as.integer(),
      ask_size2_max = max(ask_size2, na.rm = T) %>% as.integer(),
      ask_size2_min = min(ask_size2, na.rm = T) %>% as.integer(),
      ask_size2_last = last(na.omit(ask_size2)) %>% as.integer(), 
      ask_size2_log_return_volatility = sqrt(sum(log((ask_size2/ask_size2_lag))**2, na.rm = T)),
      
      price_start = first(na.omit(price)),
      price_max = max(price, na.rm = T),
      price_min = min(price, na.rm = T),
      price_last = last(na.omit(price)), 
      price_log_return_volatility = sqrt(sum(log((price/price_lag))**2, na.rm = T)),
      
      size_start = first(na.omit(size)) %>% as.integer(),
      size_max = max(size, na.rm = T) %>% as.integer(),
      size_min = min(size, na.rm = T) %>% as.integer(),
      size_last = last(na.omit(size)) %>% as.integer(), 
      size_log_return_volatility = sqrt(sum(log((size/size_lag))**2, na.rm = T)),
      
      order_count_start = first(na.omit(order_count))%>% as.integer(),
      order_count_max = max(order_count, na.rm = T) %>% as.integer(),
      order_count_min = min(order_count, na.rm = T) %>% as.integer(),
      order_count_last = last(na.omit(order_count)) %>% as.integer(),     
      order_count_log_return_volatility = sqrt(sum(log((order_count/order_count_lag))**2, na.rm = T)),
      
      Bid1_Ask1_Spread_start = first(na.omit(Bid1_Ask1_Spread)),
      Bid1_Ask1_Spread_max = max(Bid1_Ask1_Spread, na.rm = T),
      Bid1_Ask1_Spread_min = min(Bid1_Ask1_Spread, na.rm = T),
      Bid1_Ask1_Spread_last = last(na.omit(Bid1_Ask1_Spread)),     
      Bid1_Ask1_Spread_log_return_volatility = sqrt(sum(log((Bid1_Ask1_Spread/Bid1_Ask1_Spread_lag))**2, na.rm = T)),

      Bid1_Ask1_Margin_start = first(na.omit(Bid1_Ask1_Margin)),
      Bid1_Ask1_Margin_max = max(Bid1_Ask1_Margin, na.rm = T),
      Bid1_Ask1_Margin_min = min(Bid1_Ask1_Margin, na.rm = T),
      Bid1_Ask1_Margin_last = last(na.omit(Bid1_Ask1_Margin)),     
      Bid1_Ask1_Margin_log_return_volatility = sqrt(sum(log((Bid1_Ask1_Margin/Bid1_Ask1_Margin_lag))**2, na.rm = T)),
      
      Bid2_Ask2_Spread_start = first(na.omit(Bid2_Ask2_Spread)),
      Bid2_Ask2_Spread_max = max(Bid2_Ask2_Spread, na.rm = T),
      Bid2_Ask2_Spread_min = min(Bid2_Ask2_Spread, na.rm = T),
      Bid2_Ask2_Spread_last = last(na.omit(Bid2_Ask2_Spread)),     
      Bid2_Ask2_Spread_log_return_volatility = sqrt(sum(log((Bid2_Ask2_Spread/Bid2_Ask2_Spread_lag))**2, na.rm = T)),
      
      Bid2_Ask2_Margin_start = first(na.omit(Bid2_Ask2_Margin)),
      Bid2_Ask2_Margin_max = max(Bid2_Ask2_Margin, na.rm = T),
      Bid2_Ask2_Margin_min = min(Bid2_Ask2_Margin, na.rm = T),
      Bid2_Ask2_Margin_last = last(na.omit(Bid2_Ask2_Margin)),     
      Bid2_Ask2_Margin_log_return_volatility = sqrt(sum(log((Bid2_Ask2_Margin/Bid2_Ask2_Margin_lag))**2, na.rm = T)),
      
      WAP1_price_Spread_max = max(WAP1_price_Spread, na.rm = T),
      WAP1_price_Spread_min = min(WAP1_price_Spread, na.rm = T),
      WAP1_price_Spread_q1 = quantile(WAP1_price_Spread, prob = 0.25, na.rm=T),
      WAP1_price_Spread_mean = mean(WAP1_price_Spread, na.rm=T),
      WAP1_price_Spread_q3 = quantile(WAP1_price_Spread, prob = 0.75, na.rm=T),
  
      WAP1_price_Margin_max = max(WAP1_price_Margin, na.rm = T),
      WAP1_price_Margin_min = min(WAP1_price_Margin, na.rm = T),
      WAP1_price_Margin_q1 = quantile(WAP1_price_Margin, prob = 0.25, na.rm=T),
      WAP1_price_Margin_mean = mean(WAP1_price_Margin, na.rm=T),
      WAP1_price_Margin_q3 = quantile(WAP1_price_Margin, prob = 0.75, na.rm=T),
      
      order_sum = sum(order_count, na.rm = T) %>% as.integer(),
      size_sum = sum(size, na.rm = T) %>% as.integer(),
      transaction_count = sum(count_BuySell) %>% as.integer(),
      transaction_amount_sum = sum(transaction_amount, na.rm = T)
      ), by=.(time_id)]
  
  
  
  book_trade_merge_summary <- merge(book_trade_merge_summary, time_memory_whole, by=c("time_id"))
  book_trade_merge_summary[,stock_id:=stock_id_]
  #####
  
  
  
  return(book_trade_merge_summary)
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
n.cores <- 15
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
  #i <- i+1; print(i)
}

parallel::stopCluster(cl = my.cluster)
toc()
# cols_python_object <- names(py$train_)
# cols_r_object <-names(train)
# 
# setdiff(cols_python_object, cols_r_object)
# setdiff(cols_r_object, cols_python_object)

train <- read_csv("input/optiver-realized-volatility-prediction/train.csv") %>% as.data.table()
train_[,stock_id:=as.character(stock_id)]

train[, stock_id:=as.character(stock_id)]
train_merge <- merge(train_, train, by=c("stock_id","time_id"), all.x = T)
saveRDS(train_merge,"./data/train.rds")
train <- readRDS("./data/train.rds") %>% as.data.table()
#train$time_id <- NULL
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
unregister_dopar()
#train <- cbind(train[,1:30], target = train$target)

train
train <- train %>% janitor::clean_names()
train[, transaction:=ifelse(transaction_amount_sum==0,"no","yes")]
Optiver_split <- rsample::initial_split(
  train, 
  prop = 0.8, 
  strata = target
)


preprocessing_recipe <- 
  recipes::recipe(target ~ ., data = train) %>%
  # step_rm(target) %>% 
  #step_string2factor(time_id) %>% 
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  #step_impute_knn(all_predictors(), neighbors = 10) %>% 
  step_dummy(stock_id, time1, time2, time3, time4, time5, time6, time7, transaction) %>% 
  # combine low frequency factor levels
  #recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  #recipes::step_nzv(all_nominal()) %>%
  prep()

Optiver_split_train <- preprocessing_recipe %>% bake(training(Optiver_split)) %>% as.data.table()
Optiver_split_train[, names(Optiver_split_train) := lapply(.SD, function(x) ifelse(is.na(x) | is.infinite(x), 0, x))]

Optiver_split_eval <- preprocessing_recipe %>% bake(testing(Optiver_split)) %>% as.data.table()
Optiver_split_eval[, names(Optiver_split_eval) := lapply(.SD, function(x) ifelse(is.na(x) | is.infinite(x), 0, x))]
#test.case <- Optiver_split_train[is.na(price_start)]


Optiver_split_train_x <- Optiver_split_train[,-c("target")] %>% as.data.frame() %>% as.matrix()
Optiver_split_train_y <- Optiver_split_train[,c("target")] %>% as.data.frame() %>% as.matrix()
Optiver_split_eval_x <- Optiver_split_eval[,-c("target")] %>% as.data.frame() %>% as.matrix()
Optiver_split_eval_y <- Optiver_split_eval[,c("target")] %>% as.data.frame() %>% as.matrix()

dtrain <- xgb.DMatrix(Optiver_split_train_x, label=Optiver_split_train_y)
deval <- xgb.DMatrix(Optiver_split_eval_x, label=Optiver_split_eval_y)

watchlist <- list(train = dtrain, eval = deval)
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1 / (1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

rmspe  <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- sqrt(mean(((labels-preds)/labels)^2))
  return(list(metric = "rmspe", value = err, higher_better=FALSE))
}

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), Optiver_split_train),
  learn_rate(),
  size = 30
) %>% as.data.table()
xgb_grid %>% setnames(c("tree_depth","min_n","loss_reduction","sample_size","learn_rate"),c("max_depth","min_child_weight","gamma","subsample","eta"))
xgb_grid[, colsample_bytree:=mtry/ncol(Optiver_split_train_x)]


i <- 19

params_input <- 
  list(max_depth = as.integer(xgb_grid[i,1]),
       min_child_weight = as.double(xgb_grid[i,2]),
       gamma = xgb_grid[i,3],
       subsample = xgb_grid[i,4],
       eta = 0.3,
       colsample_bytree = as.double(xgb_grid[i,7])
  )
bst <- xgb.train(dtrain, 
                 nthread = 15, nrounds = 1000, params = params_input, watchlist, feval = rmspe, objective = "reg:squarederror",verbose = 1, save_name = "./data/xgb_model",early_stopping_rounds = 15, maximize = F
                 )

bst_importance <- xgb.importance(colnames(Optiver_split_train[,-c("target")]), model = bst)

Optiver_cv_folds <- 
  # recipes::bake(
  #   preprocessing_recipe, 
  #   new_data = training(Optiver_split)
  # ) %>%  
  rsample::vfold_cv(Optiver_split_train, v = 5, strata = target)

xgboost_model <- 
  parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    tree_depth = tune(), min_n = tune(), 
    loss_reduction = tune(),                     ## first three: model complexity
    sample_size = tune(), 
    #mtry = tune(),         ## randomness
    learn_rate = tune(),  
  ) %>%
  set_engine("xgboost", objective = "reg:squarederror")


xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  #finalize(mtry(), Optiver_split_train),
  learn_rate(),
  size = 30
)


xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>% 
  add_formula(target ~ .)

tic()
doParallel::registerDoParallel(cl = my.cluster)
xgboost_tuned <- tune::tune_grid(
  object = xgboost_wf,
  resamples = Optiver_cv_folds,
  grid = xgb_grid,
  metrics = yardstick::metric_set(rmse, rsq, mae),
  control = tune::control_grid(verbose = TRUE)
)

parallel::stopCluster(cl = my.cluster)
toc()