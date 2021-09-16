
train <- readRDS("./data/train.rds") %>% as.data.table()
#train$time_id <- NULL
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
unregister_dopar()
#train <- cbind(train[,1:30], target = train$target)


train <- train %>% janitor::clean_names()
train[, transaction:=ifelse(transaction_amount_sum==0,"no","yes")]
train_distilled <- train[,.(stock_id,time_id,wap1_log_return_volatility,wap2_log_return_volatility)]

time_ids <- train$time_id %>% unique()
stock_ids <- train$stock_id %>% unique()
wap_log_return <- function (stock_id, time_id, input){
  train_distilled[stock_id!=setforloop[i,stock_id] & time_id==setforloop[i,time_id],][,list(wap1_log_retutn_volatility_mean_etc = mean(wap1_log_return_volatility))]
}
train_distilled[time_id==5]
setforloop <- train_distilled[, .(stock_id,time_id)]

parallel::detectCores()
n.cores <- 15
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
print(my.cluster)

tic()
doParallel::registerDoParallel(cl = my.cluster)

for (time_id_input in time_ids[1:2]){
  train_distilled_sel_by_time_id <- train_distilled[time_id==time_id_input]
  stock_ids <- train_distilled_sel_by_time_id$stock_id %>% unique()
  
  wap_log_return <- foreach(
    stock_id_input = stock_ids[1:10],
    .combine = "rbind",
    .packages = c("tidyverse","data.table")
  ) %dopar% {
    train_distilled_sel_by_time_id_stock_id <- train_distilled_sel_by_time_id[stock_id != stock_id_input][,list(wap1_log_retutn_volatility_mean_etc = mean(wap1_log_return_volatility))]
    train_distilled_sel_by_time_id_stock_id[, time_id:=time_id_input]
    train_distilled_sel_by_time_id_stock_id[, stock_id:=stock_id_input]
    #i <- i+1; print(i)
  }
}

unregister_dopar()
parallel::stopCluster(cl = my.cluster)
toc()
