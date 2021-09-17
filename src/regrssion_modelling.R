
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
n.cores <- 16
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
print(my.cluster)

tic()
doParallel::registerDoParallel(cl = my.cluster)
wap_log_return_whole <- data.table()
for (time_id_input in time_ids){
  train_distilled_sel_by_time_id <- train_distilled[time_id==time_id_input]
  stock_ids <- train_distilled_sel_by_time_id$stock_id %>% unique()
  
  wap_log_return <- foreach(
    stock_id_input = stock_ids,
    .combine = "rbind",
    .packages = c("tidyverse","data.table")
  ) %dopar% {
    train_distilled_sel_by_time_id_stock_id <- 
      train_distilled_sel_by_time_id[stock_id != stock_id_input][,list(wap1_log_retutn_volatility_other_stock_ids_mean = mean(wap1_log_return_volatility),
                                                                       wap1_log_retutn_volatility_other_stock_ids_sd = sd(wap1_log_return_volatility),
                                                                       wap2_log_retutn_volatility_other_stock_ids_mean = mean(wap2_log_return_volatility),
                                                                       wap2_log_retutn_volatility_other_stock_ids_sd = sd(wap2_log_return_volatility))]
    train_distilled_sel_by_time_id_stock_id[, time_id:=time_id_input]
    train_distilled_sel_by_time_id_stock_id[, stock_id:=stock_id_input]
    #i <- i+1; print(i)
  }
  wap_log_return_whole <- rbind(wap_log_return_whole,wap_log_return)
}

unregister_dopar()
parallel::stopCluster(cl = my.cluster)
toc()

train_for_regression <- merge(train[,.(wap1_log_return_volatility,wap2_log_return_volatility,time_id,stock_id)], wap_log_return_whole, by=c("time_id","stock_id"))
train_for_regression[,stock_id:=as.double(stock_id)]
train_target <- read_csv("input/optiver-realized-volatility-prediction/train.csv") %>% as.data.table()

train_for_regression_ <-  merge(train_for_regression, train_target, by=c("time_id","stock_id"))

train_for_regression_$time_id <- NULL
train_for_regression_[, stock_id:=as.character(stock_id)]
Optiver_split <- rsample::initial_split(
  train_for_regression_, 
  prop = 0.8, 
  strata = target
)

preprocessing_recipe <- 
  recipes::recipe(target ~ ., data = train_for_regression_) %>%
  # step_rm(target) %>% 
  #step_string2factor(time_id) %>% 
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  #step_impute_knn(all_predictors(), neighbors = 10) %>% 
  step_dummy(stock_id) %>% 
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


i <- 9

params_input <- 
  list(max_depth = as.integer(xgb_grid[i,1]),
       min_child_weight = as.double(xgb_grid[i,2]),
       gamma = xgb_grid[i,3],
       subsample = xgb_grid[i,4],
       eta = 0.3,
       colsample_bytree = as.double(xgb_grid[i,7])
  )
bst <- xgb.train(dtrain, 
                 nthread = 15, nrounds = 1000, params = params_input, watchlist, feval = rmspe, objective = "reg:squarederror",verbose = 1, 
                 save_name = "./data/xgb_model",early_stopping_rounds = 15, maximize = F
)

#xgb.save(bst, "xgbmodel_Optiver")
bst <- xgb.load("xgbmodel_Optiver")
train_for_regression_wo_stock_ids <- train_for_regression_[,-c("stock_id")]

fm0 <- lm(target ~ ., train_for_regression_wo_stock_ids)
X <- model.matrix(fm0)
f <- function(b) with(train_for_regression_wo_stock_ids, sqrt(mean(((X-b)/b)^2)))
res <- optim(coef(fm0), f, method = "BFGS")
res

train_for_regression_[,prediction:=fm0$coefficients[1]+fm0$coefficients[2]*wap1_log_return_volatility+fm0$coefficients[3]*wap2_log_return_volatility+
                        fm0$coefficients[4]*wap1_log_retutn_volatility_other_stock_ids_mean+
                        fm0$coefficients[5]*wap1_log_retutn_volatility_other_stock_ids_sd+
                        fm0$coefficients[6]*wap2_log_retutn_volatility_other_stock_ids_mean+
                        fm0$coefficients[7]*wap2_log_retutn_volatility_other_stock_ids_sd]
train_for_regression_[,list(result=sqrt(mean((target-prediction)/target)^2))]

Optiver_split_eval <- preprocessing_recipe %>% bake(testing(Optiver_split)) %>% as.data.table()
Optiver_split_eval_x <- Optiver_split_eval[,-c("target")] %>% as.data.frame() %>% as.matrix()
prediction_xgb <- predict(bst, newdata = Optiver_split_eval_x)
train_for_regression_wo_stock_ids_xgb <- testing(Optiver_split) %>% cbind(prediction_xgb)
train_for_regression_wo_stock_ids_xgb[,list(result=sqrt(mean((target-prediction_xgb)/target)^2))]

train_for_regression_wo_stock_ids_xgb[, ensemble:=0.949*prediction_xgb] # result 0.002891952
train_for_regression_wo_stock_ids_xgb[,list(result=sqrt(mean((target-ensemble)/target)^2))]

bst
ggplot(train_for_regression_wo_stock_ids_xgb) + 
  geom_point(aes(x=prediction_xgb, y=target))
