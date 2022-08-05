function [F,g,a,b,a_data] = Linear_Exponential_Smoothing(n_x,a,b,a_data,forecasting,estimated_state)
%% This function performs Linear Exponential Smoothning to assess state
%% transition matrix F and vector g
% 
alfa=0.9*ones(n_x,1);
beta=0.4*ones(n_x,1);

% alfa=0.8*ones(n_x,1);
% beta=0.5*ones(n_x,1);

% mentioned in the paper but it will cause low frequency deviation of the
% prediction values, maybe that's because of EKF and UKF
% alfa=0.001*ones(n_x,1);
% beta=2*ones(n_x,1);

F=zeros(n_x);
for i_F=1:n_x
    F(i_F,i_F)=alfa(i_F)*(1+beta(i_F));
end

g=((1-alfa).*(1+beta)).*forecasting+(1-beta).*b-beta.*a;
  a=alfa.*estimated_state+(1-alfa).*forecasting;
  a_data=[a_data a];
  b=beta.*(a-a_data(:,end-1))+(1-beta).*b;
  
end
