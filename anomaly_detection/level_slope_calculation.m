function [a, b] = level_slope_calculation(state_data_filtered)
% Program for calculation level and slope component for forecasted value (Holt technique)
  % This program calculates level and slope for initial forecasting based on
  % tecnnique explained in 'Forecasting Trends_Exponential Smoothing', pp 31

last_observations = state_data_filtered(:,(end-3):(end-1));

a = (last_observations(:,1)+last_observations(:,2)+last_observations(:,3))/3+(last_observations(:,3)-last_observations(:,1))/2;
b = (last_observations(:,3)-last_observations(:,1))/2;

end

