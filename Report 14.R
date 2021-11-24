# 0) Prelúdio =======

setwd('/home/heitor/Área de Trabalho/R Projects/Análise Macro/Labs/Lab 14')

library(tidyverse)   # padrão manipulação e visualização
library(tidymodels)  # padrão ML
library(glmnet)      # Elastic Net
library(plsmod)      # Partial Least Squares
library(plotly)      
library(ggExtra)     # plots marginais
library(GGally)
library(ISLR)

dds <- as.data.frame(Hitters) %>%
	as_tibble() %>% na.omit()

# 1) Estatísticas Descritivas =======

dds %>% summary()

ggcorr(dds %>%
	   	select(!c('NewLeague', 'League', 'Division')),
	   label = T)

ggplot(dds, aes(x=Salary)) +
	geom_histogram(aes(y=..density..), bins = 30) + 
	geom_density() +
	geom_vline(aes(xintercept = Salary %>%
				   	mean(na.rm=T)),
			   linetype="dashed")

ggplot(dds, aes(x=Errors)) +
	geom_histogram(aes(y=..density..), bins = 30) + 
	geom_density() +
	geom_vline(aes(xintercept = Errors %>%
				   	mean(na.rm=T)),
			   linetype="dashed")

ggplotly(
ggplot(dds, aes(y=Salary, x=Runs)) +
	geom_point(aes(color=Division))+
	geom_density_2d())

ggplotly(
ggplot(dds, aes(y=Salary, x=Errors)) +
	geom_point(aes(color=Division)) +
	geom_density_2d(alpha=.5,
					color='darkviolet'))

ggplotly(
ggplot(dds, aes(y=Salary, x=Hits)) +
	geom_point(aes(color=Division)) +
	geom_density_2d(alpha=.5,
					color='darkviolet'))

ggplotly(
ggplot(dds, aes(y=Salary, x=Years)) +
	geom_point(aes(color=Division))+
	geom_density_2d(alpha=.5,
					color='darkviolet'))

ggplotly(
	ggplot(dds, aes(y=CHits, x=Hits)) +
		geom_point(aes(color=Division)) )

# 2) Divisão: Treino & Teste =======

slice_1   <- initial_split(dds)
train_dds <- training(slice_1)
test_dds  <- testing(slice_1)

# 3) Modelo =======

algorit_elast <- linear_reg(
	penalty = tune::tune(),
	mixture = tune::tune()) %>%
	set_mode('regression') %>% 
	set_engine("glmnet")
algorit_elast %>% translate()

algorit_pls <- pls(num_comp = 7) %>% 
	set_mode("regression") %>% 
	set_engine("mixOmics")

# 4) Fórmula =======

formula_geral <- recipe(Salary~.,
						data = train_dds) %>% 
	step_normalize(all_numeric_predictors()) %>% 
	step_dummy(all_nominal_predictors()) %>%
	prep()

formula_geral %>% bake(new_data=NULL)

#formula_mqp <- recipe(Salary~.,
#					  data = train_dds) %>% 
#	step_normalize(all_numeric_predictors()) %>% 
#	step_dummy(all_nominal_predictors()) %>%
#	step_pls(all_numeric_predictors(), 
#			 outcome = "Salary",
#			 num_comp = tune())
#
# Eu não farei outro recipe com o pls, mas outro modelo  com o pls: o modelo já tem tune(), se eu fizesse outro recipe, teria tune, eu aplicaria outro tune() em cima, não quero isso.

# 5) Workflow =======

wrkflw_elast <- workflow() %>%
	add_model(algorit_elast) %>%
	add_recipe(formula_geral)

wrkflw_pls <- workflow() %>%
	add_model(algorit_pls) %>%
	add_recipe(formula_geral)

wrkflw_pls <- wrkflw_pls %>% 
	last_fit(slice_1)

# 6) Validação =======

kfold_geral <- vfold_cv(train_dds,
					 v=10,
					 repeats = 2)

# 7) Ajustes e Treinamentos =======

## limites do tune() ---

grid_padr <- wrkflw_elast %>%
	parameters() %>% 
	update(
		penalty = penalty(range = c(.25, .75)),
		mixture = mixture(range = c(0, 1))
	) %>% 
	grid_regular(levels = 5)

## afinação dos tune() ---

afinç_cv_elast <- wrkflw_elast %>% 
	tune_grid(resamples = kfold_geral,
			  grid      = grid_padr,
			  control   = control_grid(save_pred = T),
			  metrics   = metric_set(rmse, mae))

afinç_cv_elast
wrkflw_pls$.metrics

# 8) Seleção do melhor modelo =======

afinç_cv_elast %>% ggplot2::autoplot()

afinç_cv_elast %>% show_best(n=1, 'rmse')
wrkflw_pls %>% show_best(n=1, 'rmse')

melhor_tune <- select_best(afinç_cv_elast, 'rmse') 

## Finalmentes ---

algorit_fnl_elast <- algorit_elast %>% 
	finalize_model(parameters = melhor_tune)

wrkflw_fnl_elast <- workflow() %>% 
	add_model(algorit_fnl_elast) %>% 
	add_recipe(formula_geral) %>% 
	last_fit(slice_1)

wrkflw_fnl_elast$.metrics
wrkflw_pls$.metrics

gg_pred <- wrkflw_fnl_elast %>%
	collect_predictions() %>%
	ggplot(aes(x=.pred, y=Salary)) +
	geom_point(alpha=.75) +
	geom_abline(slope     = 1,
				intercept = 0,
				color     = 'palegreen4',
				size      =.8)

ggExtra::ggMarginal(gg_pred,
					type    = 'density',
					margins = 'both',
					colour  = 'palegreen4',
					fill    = 'palegreen1')


gg_pred_pls <- wrkflw_pls %>%
	collect_predictions() %>%
	ggplot(aes(x=.pred, y=Salary)) +
	geom_point(alpha=.75) +
	geom_abline(slope     = 1,
				intercept = 0,
				color     = 'salmon4',
				size      =.8)

ggExtra::ggMarginal(gg_pred_pls,
					type    = 'density',
					margins = 'both',
					colour  = 'salmon4',
					fill    = 'salmon')
#########################################################
#terms_pls <- c(2:10)
#
#for (t_pls in terms_pls) {
#	algorit_pls <- pls( X = formula_geral %>%
#							bake(new_data=NULL) %>% 
#							dplyr::select(-Salary),
#						Y = formula_geral %>%
#							bake(new_data=NULL) %>%
#							dplyr::select(Salary),
#		ncomp = as.numeric(t_pls)) %>% 
#		set_mode("regression") %>% 
#		set_engine("mixOmics")
#	wrkflw_pls <- workflow() %>%
#		add_model(algorit_pls) %>% 
#		last_fit(slice_1)
#	wrkflw_pls %>% show_best(n=1, 'rmse')
#}
#
# Não deu certo!!! :'(