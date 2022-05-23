source("setting.R")
library(metR)
library(Rsolnp)
set.seed(42)

# fs::dir_delete("output/")
fs::dir_create("output/")
# 理論的な関係 ------------------------------------------------------------------

f <- function(ds, de, biz) {
  ds * de^0.6 * biz^0.8 / 20
}

df_master <- crossing(
  ds = 0:100,
  de = 0:100,
  biz = 0:100
) %>%
  mutate(income = f(ds, de, biz))


# 3次元は見ずらいのでDE力は固定
title = "スキルレベルと年収の関係（DE力を50で固定）"
df_master %>%
  filter(de == 50) %>%
  ggplot(aes(ds, biz)) +
  geom_contour2(
    aes(z = income, label = ..level.., color = ..level..),
    breaks = AnchorBreaks(1000, binwidth = 200),
    family = base_family
  ) +
  scale_x_continuous(breaks = breaks_width(20), limits = c(0, 100)) +
  scale_y_continuous(breaks = breaks_width(20), limits = c(0, 100)) +
  scale_color_viridis_c() +
  labs(
    x = "データサイエンス力",
    y = "ビジネス力",
    title = title
  ) +
  theme_scatter()

save_plot(last_plot(), glue("output/{title}.png"))

# シミュレーションデータの生成 ----------------------------------------------------------

N <- 2000
df <- tibble(
  ds = rbeta(N, 13, 7) * 100,
  de = rbeta(N, 10, 10) * 100,
  biz = rbeta(N, 7, 13) * 100,
  income = f(ds, de, biz) * rnorm(N, 1, 0.05)
)


# シミュレーションデータの可視化 ---------------------------------------------------------

title = "年収の分布"
df %>%
  ggplot(aes(income)) +
  geom_histogram(binwidth = 100, fill = cols[8], alpha = 0.8) +
  scale_x_continuous(breaks = breaks_width(200)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    x = "年収",
    y = "度数",
    title = title
  ) +
  theme_line()

save_plot(last_plot(), glue("output/{title}.png"))

# 可視化用に縦持ちに変える
df_longer <- df %>%
  pivot_longer(c(ds, de, biz)) %>%
  mutate(name = factor(name, levels = c("ds", "de", "biz"), labels = c("DS", "DE", "Biz")))

title = "スキルレベルの分布"
df_longer %>%
  ggplot(aes(value, fill = name)) +
  geom_histogram(position = position_identity(), binwidth = 5, alpha = 0.5) +
  scale_x_continuous(breaks = breaks_width(20), limits = c(0, 100)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(name = "", values = cols[c(1, 3, 5)]) +
  labs(
    x = "スキルレベル",
    y = "度数",
    title = title
  ) +
  theme_line()

save_plot(last_plot(), glue("output/{title}.png"))


title = "スキルレベルと年収の関係"
df_longer %>%
  ggplot(aes(value, income, color = name)) +
  geom_point(alpha = 0.3) +
  geom_smooth(color = "gray") +
  facet_wrap(~name) +
  scale_x_continuous(
    breaks = breaks_width(20),
    limits = c(0, 100),
    expand = expansion(mult = c(0.15))
  ) +
  scale_y_continuous(breaks = breaks_width(200)) +
  scale_color_manual(name = "", values = cols[c(1, 3, 5)]) +
  labs(
    x = "スキルレベル",
    y = "年収",
    title = title
  ) +
  theme_scatter()

save_plot(last_plot(), glue("output/{title}.png"))


# データの分割 ------------------------------------------------------------------

train_test_split <- rsample::initial_split(df)
df_train <- rsample::training(train_test_split)
df_test <- rsample::testing(train_test_split)
cv_split <- rsample::vfold_cv(df_train, v = 5)

# モデルの定義 -----------------------------------------------------------

model <- parsnip::linear_reg(
  penalty = tune::tune(),
  mixture = tune::tune()
) %>%
  parsnip::set_mode("regression") %>%
  parsnip::set_engine("glmnet")

# 前処理 ---------------------------------------------------------------------


# 目的変数の対数だけとる
rec <- recipes::recipe(income ~ ds + de + biz, data = df) %>%
  recipes::step_interact(terms = ~ ds:de + de:biz + ds:biz + ds:de:biz) %>%
  recipes::step_poly(all_predictors(), degree = 4) %>%
  recipes::step_normalize(all_numeric_predictors())

# ワークフローの作成-------------------------------------------------------------------------

# 前処理をしたrecipeがあるならそれもadd_recipe()で追加する
wf <- workflows::workflow() %>%
  workflows::add_model(model) %>%
  workflows::add_recipe(rec)



# モデルのチューニング --------------------------------------------------------------

# ハイパーパラメータの探索
# tune_bayes()ならベイズ最適化、tune_grid()でグリッドサーチもできる
bayes_result <- wf %>%
  tune::tune_bayes(
    resamples = cv_split,
    param_info = tune::parameters(
      list(
        penalty = dials::penalty(),
        mixture = dials::mixture()
      )
    ),
    metrics = yardstick::metric_set(rmse), # RMSEが一番良くなるパラメータを探す
    initial = 5,
    iter = 30,
    control = tune::control_bayes(verbose = TRUE, no_improve = 5)
  )


bayes_result

# 予測精度を確認
bayes_result %>%
  tune::collect_metrics()

# 一番良かったハイパーパラメータでの予測精度を確認
bayes_result %>%
  tune::show_best()

# 一番良かったモデルを抽出
best_model <- bayes_result %>%
  tune::select_best()

# ワークフローをアップデート
wf_final <- wf %>%
  tune::finalize_workflow(best_model)



# 最終モデルの作成 ----------------------------------------------------------------

# 全訓練データで学習して、テストデータで予測
final_result <- wf_final %>%
  tune::last_fit(train_test_split)

# 予測精度を確認
final_result %>%
  tune::collect_metrics()


plot_pred_actual <- function(df, actual_var, title) {
  lims <- extendrange(pull(df, {{ actual_var }}))
  df %>%
    ggplot(aes(x = {{ actual_var }}, y = .pred)) +
    geom_abline(color = cols[7], size = 1) +
    geom_point(color = cols[2], alpha = 0.5) +
    coord_fixed(xlim = lims, ylim = lims) +
    scale_x_continuous(breaks = breaks_width(200)) + 
    scale_y_continuous(breaks = breaks_width(200)) + 
    labs(x = "実測", y = "予測", title = title) +
    theme_scatter()
}

title = "実測値と予測値の比較"
final_result %>%
  tune::collect_predictions() %>%
  plot_pred_actual(income, title)

save_plot(last_plot(), glue("output/{title}.png"))


# 最終モデルを出力
model <- final_result %>%
  tune::extract_workflow()


# Counterfactual Expampleを生成 -------------------------------------------------------

counterfactual_explanations <- function(model, current_x, desired_y) {
  as_input <- function(x) tibble(ds = x[1], de = x[2], biz = x[3]) # ベクトルをモデルの入力に
  predict_num <- function(model, x) pull(predict(model, as_input(x))) # 予測結果をdfからnumericに
  constraint <- function(x) predict_num(model, x) - desired_y # 制約条件
  distance <- function(x) norm(current_x - x, type = "2") # 目的関数

  solution <- Rsolnp::solnp( # 最適化関数
    pars = current_x + 1e-3,
    fun = distance,
    ineqfun = constraint,
    ineqLB = 0,
    ineqUB = 0.1,
    LB = current_x,
    UB = c(100, current_x[2] + 1e-2, 100), # DE力は鍛えないことにする
    control = list(tol = 1e-5)
  )

  result <- list(
    current_x = as_input(current_x), # 現状のスキル
    current_y = predict_num(model, current_x), # 現状の予測年収
    desired_y = desired_y, # 達成したい年収
    required_x = as_input(solution$pars), # 必要なスキルレベル
    predicted_y = predict_num(model, solution$pars) # 達成された予測年収
  )
  return(result)
}


df_instance <- df %>%
  mutate(income_pred = pull(predict(model, .))) %>%
  filter(income_pred %>% between(600, 610)) %>%
  sample_n(1)

df_instance

x_instance <- df_instance %>%
  select(ds, de, biz) %>%
  as.numeric()

result <- counterfactual_explanations(
  model,
  current_x = x_instance,
  desired_y = 1000
)

result


df_pred <- crossing(
  ds = 10:90,
  de = result$required_x$de,
  biz = 10:90
) %>%
  mutate(income = pull(predict(model, .)))

title = "Counterfactualの可視化"
df_pred %>%
  ggplot(aes(ds, biz)) +
  geom_point(data = df, color = cols[7], alpha = 0.2) +
  geom_contour2(
    aes(z = income, label = ..level.., color = ..level..),
    breaks = AnchorBreaks(1000, binwidth = 200),
    family = base_family
  ) +
  geom_point(
    data = result$current_x,
    shape = 21,
    size = 4,
    color = "white",
    fill = cols[2]
  ) +
  geom_point(
    data = result$required_x,
    shape = 21,
    size = 4,
    color = "white",
    fill = cols[6]
  ) +
  geom_segment(
    aes(
      x = result$current_x$ds + 1,
      y = result$current_x$biz + 1,
      xend = result$required_x$ds - 1,
      yend = result$required_x$biz - 1
    ),
    size = 1,
    arrow = arrow(length = unit(0.2, "cm")),
    color = cols[9]
  ) +
  scale_x_continuous(breaks = breaks_width(20), limits = c(20, 90)) +
  scale_y_continuous(breaks = breaks_width(20), limits = c(10, 70)) +
  scale_color_viridis_c() +
  labs(
    x = "データサイエンス力",
    y = "ビジネス力",
    title = title
  ) +
  theme_scatter()

save_plot(last_plot(), glue("output/{title}.png"))
