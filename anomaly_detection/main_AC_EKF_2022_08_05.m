% Shared version 2022 - August - 05
% WLS and EKF state estimation, plus FDI and SLC
% The power flow has been done using OPF via matpower (version 7.0)
%%%%%%%%%%%%%%%%%% Important %%%%%%%%%%%%%%%%%%%
% In order to be able to run this code you must have matpower installed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all
format short g
%
disp(' ')
disp(' ============== START AC SE ================ ')
disp(' =========================================== ')
disp(' ')

% first loop is to select SLC bus
for www = 14%[2:6 8:14] % bus 1 and 7 have zero load
    % second loop is to select percentage of SLC    
    for ww = 0.5%0.1:0.1:1 % amount of SLC 
%%%%%%%%%%%%%%%%
load_profile = linspace(1,0.95,500);
T = numel(load_profile); % time series
B_N = 14;                     % Bus number
num_meas = 80; %51; %41;                % measurement number
Var_N = 2*B_N - 1;            % slack bus has been excluded
V_WLS_true = zeros(B_N,T);   % bus 1 is slack bus
V_WLS_noisy = zeros(B_N,T);
theta_WLS_true = zeros(B_N,T);   % bus 1 is slack bus
theta_WLS_noisy = zeros(B_N,T);
meas_T_true = zeros(num_meas,T);

%%% Sudden load change (SLC) application
SLC = 1; % "1" means we have sudden load change and "0" means we don't
SLC_bus = www;%combos(www,:);%14;%[11 12];%wwww;%input('SLC target bus: ');   % bus num that would have SLC
SLC_weight = ww; % precentage of load cut-off, 0.5 == 50%
int_slc = 200;
end_slc = 300;
if SLC == 1
    disp(' ');
    disp([' Sudden load change on bus number #', num2str(SLC_bus), ' will be implemented']);
    disp([' ',num2str(SLC_weight*100),' percent of the load will be cut-off'])
    disp(' ');
end

mpc = loadcase('case14.m'); % matpower case for IEEE 14 bus system
mpopt = mpoption('out.all', 0, 'verbose', 0); % matpower settings
unsuccess_count = 0;

for i = 1 : T
    
    mpc_temp = mpc; % just a temproray variable to be able to change load
    mpc_temp.bus(:,3) = mpc_temp.bus(:,3)*load_profile(i);
    mpc_temp.bus(:,4) = mpc_temp.bus(:,4)*load_profile(i);
    
    %%% SLC
    if SLC && (i >= int_slc && i <= end_slc)
        mpc_temp.bus(SLC_bus,3) = mpc_temp.bus(SLC_bus,3)*(1-SLC_weight);
        mpc_temp.bus(SLC_bus,4) = mpc_temp.bus(SLC_bus,4)*(1-SLC_weight);
    end
    
    res = runopf(mpc_temp,mpopt); % output of OPF
    if res.success == 0 % 0 means the opf has been unsuccessful
        unsuccess_count=unsuccess_count+1;
    end
    
    %%% just a comment on command window to follow the process every 100 iteration
    if mod(i,100) == 0
        disp([' Matpower AC OPF for sample time: ', num2str(i)]);% time instant
        disp([' And number of unseccessful OPF till now are: ', num2str(unsuccess_count)]);  % unsuccessful OPF
        disp(' ')
    end
    
    %%% voltage and phase angle obtained by OPF via matpower
    V = res.bus(:,8);
    del = res.bus(:,9)*pi/180;
    % by inserting x (voltage and phase angles) in function h (here it is wls_z)
    % we can have the measurements
    meas_T_true(:,i) = wls_z(B_N,V,del);
end

%%% noise of the system
accRT = 1; % for all measurements the same accuracy is considered
noise = 1 + (accRT/300)*randn(num_meas,T);
meas_T_noisy = meas_T_true.*noise;
z = zdatas(B_N);  % measurement data file which is inline with meas_T and used to generate H matrix
sig = accRT*meas_T_true/300;

%%% Initialization -- may help to speed up the simulation
Residuals_WLS_noisy = zeros(num_meas,T);
L2_norm = zeros(1,T);
Residuals_WLS_true = zeros(num_meas,T);
Norm_Residuals_WLS_noisy = zeros(num_meas,T);
Norm_Residuals_WLS_true = zeros(num_meas,T);
Obj_valu_WLS_noisy = zeros(num_meas,T);
Obj_valu_WLS_true = zeros(num_meas,T);
omeg = zeros(num_meas,T);

%%% fals data injection (FDI) attack vector
FDI = 1; % "1" means we have FDI and "0" means we don't
FDI_sen = 1; % FDI scenario num can be from 1 to 6
att_w = 0.1;%0.03; % attack weight
FDIA_state = www;%combos(www,:);%[8 10];%www; %input('FDIA target bus: '); % attacked bus
int_att = 350;%500; % initial point of attack
end_att = 1000; % finishing point of the attack

if FDI == 1
    disp(' ');
    disp([' Attack scenario #', num2str(FDI_sen), ' will be implemented']);
    disp(' ');
end

%%% Bad data which is caused by measurement errors
BDD = 0;
int_bdd = 50;
end_bdd = 150;
bdd_meas = 15; % measurement #15
bdd_weight = 0.2; % percentage of increase in measurement reading value

if BDD == 1
    meas_T_noisy(bdd_meas,int_bdd:end_bdd) = (1+bdd_weight)*meas_T_noisy(bdd_meas,int_bdd:end_bdd);
end

%% WLS
disp(' ')
disp(' ============== WLS ================ ')
disp(' ')
for i = 1 : T
    
    %%% check for pesudomeasurements or SLC, 
    % when load will be zero, the sigma = 0 --> sigma inverse would be inf
    % this will cause sigularity in the code so we need to remove it
    if sum(abs(sig(:,i)) < 1e-8)~=0
        sig(abs(sig(:,i))< 1e-8,i) = 10^-4;
    end
    
    if mod(i,100) == 0
        disp(['WLS at time step: ', num2str(i)]);  % time instant
    end
   
    %%% sudden load change
    if SLC == 1 && (i >= int_slc && i <= end_slc)
        SLC_check = 1;
    else
        SLC_check = 0;
    end
    
    if FDI == 0 || (i < int_att || i > end_att)
        attack_vec_WLS = zeros(num_meas,1);
    elseif FDI == 1 && FDI_sen ==1

%%% Senario 1: does have the perfect knowledge (specially about the noise)
        x_vec = [zeros(13,1);zeros(14,1)];
        x_vec(FDIA_state + 13) = att_w;
        [~, V, del] = wls(B_N, meas_T_noisy(:,i), sig(:,i));
        attack_vec_WLS = -wls_z(B_N,V,del)+wls_z(B_N,V+x_vec(B_N : end),del+[0;x_vec(1 : B_N-1)]); 
%%% more scenarios can be added
    end
    attack_vec_plot_WLS(:,i) = attack_vec_WLS;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    W = diag(sig(:,i).^-2); % W = R^-1
    
    %%% WLS state estimation (SE) 
    %Objective Function.. J = sum(inv(Ri)*r.^2); 
    
    [r, V, del] = wls(B_N, meas_T_true(:,i), sig(:,i));
    V_WLS_true(:,i) = V;
    theta_WLS_true(:,i) = del;
   
    % without noise and attack
    Residuals_WLS_true(:,i) = r; 
    Norm_Residuals_WLS_true(:,i) = r./sig(:,i);%./sqrt(abs(omeg(:,i)));
    Obj_valu_WLS_true(:,i) = W*(r).^2;
    
    % with noise and attack
    [r, V, del] = wls(B_N, meas_T_noisy(:,i) + attack_vec_WLS, sig(:,i));
    V_WLS_noisy(:,i) = V;
    theta_WLS_noisy(:,i) = del;%((H'*W*H)^-1)*H'*W*(meas_T_noisy(:,i) + attack_vec_WLS);
    Residuals_WLS_noisy(:,i) = r;%meas_T_noisy(:,i) + attack_vec_WLS - H*theta_WLS_noisy(:,i);
    Norm_Residuals_WLS_noisy(:,i) = r./sig(:,i);%./sqrt(abs(omeg(:,i)));
    L2_norm(:,i) = sqrt(sum(r.^2));
    Obj_valu_WLS_noisy(:,i) = W*(r).^2;
end


%% EKF
disp(' ')
disp(' ============== EKF ================ ')
disp(' ')

%%% Initializing 
nx = Var_N; % Number of states
Rz = diag(sig(:,4).^2); % R
%%% the H has obtained in a way that the state variable will be as follows
%%% del = x(1 : B_N - 1) and V = x(B_N : end)
H = H_matrix(B_N,V_WLS_noisy(:,4),theta_WLS_noisy(:,4));
P = inv(H'*inv(Rz)*H);

qc = -8;
Q = (10^qc)*eye(nx);
state_data_filtered = [theta_WLS_noisy(2:end,1:4);V_WLS_noisy(:,1:4)]; % it is the first 4 time sample data of true state estimation
estimated_state = [theta_WLS_noisy(2:end,4);V_WLS_noisy(:,4)];
[a, b] = level_slope_calculation(state_data_filtered); % initializing parameters for Linear Exponential Smoothing (LES) technique 
a_data = a;                   % database for 'a'
forecasting = a + b;          % initial state predictions   

n_dyn = 4;
n_dyn_2 = T;%5720;             % EKF stops at this time step
SE_data = [];
Fo_data = [];
Anomaly_mod = [];
innovation = zeros(num_meas,T - n_dyn);
Residuals_EKF = zeros(num_meas,T - n_dyn);
Norm_Residuals_EKF = zeros(num_meas,T - n_dyn);
Norm_innovation = zeros(num_meas,T - n_dyn);
Norm_innovation2 = zeros(num_meas,T - n_dyn);
Norm_res_paper = zeros(Var_N,T - n_dyn);

        for j = (n_dyn + 1) : n_dyn_2
            Rz = diag(sig(:,j).^2);
            
            %%% sudden load change
            if SLC == 1 && (j >= int_slc && j <= end_slc)
                SLC_check = 1;
            else
                SLC_check = 0;
            end
            
            if FDI == 0 || j < int_att || j > end_att
                attack_vec_EKF = zeros(num_meas,1);
            elseif FDI == 1 && FDI_sen == 1
        
        %%% Senario 1: does have the perfect knowledge (specially about the noise)
                x_vec = [zeros(13,1);zeros(14,1)];
                x_vec(FDIA_state + 13) = att_w;
                [~, V, del] = wls(B_N, meas_T_noisy(:,j), sig(:,j));
                attack_vec_EKF = -wls_z(B_N,V,del) + wls_z(B_N,V + x_vec(B_N : end), del+[0; x_vec(1 : B_N-1)]);
        %%% more scenarios can be added
        
            end
            attack_vec_plot_EKF(:,j) = attack_vec_EKF;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            if mod(j,100) == 0
                disp(['EKF time instant is: ', num2str(j)]);  % time instant
            end
            %%% Forecasting step (Time update)
            %%% LES
            [F, g, a, b, a_data] = Linear_Exponential_Smoothing(nx, a, b, a_data, forecasting, estimated_state);
            forecasting = F*estimated_state + g;% + diag(Q); % predicted state vector
        
            H = H_matrix(B_N,forecasting(B_N : end),[0;forecasting(1:B_N - 1)]);
            M = F*P*F' + Q;     % state Prediction Error Covariance Matrix (P-)
            S = H*M*H' + Rz;    % Measurement prediction-error (or Innovation) covariance matrix
            K = M*H'/S;         % Kalman gain
            P = M - K*H*M;      % state estimation Error Covariance Matrix (P+)
        
            %%% Filtering step
            z_for_EKF = wls_z(B_N,forecasting(B_N : end),[0;forecasting(1:B_N - 1)]);
            estimated_state = forecasting + K*(meas_T_noisy(:,j) + attack_vec_EKF - z_for_EKF);
        
            z_se_EKF = wls_z(B_N,estimated_state(B_N : end),[0;estimated_state(1:B_N - 1)]); 
            innovation(:,j - n_dyn) = meas_T_noisy(:,j) + attack_vec_EKF - z_for_EKF;
            Norm_innovation(:,j - n_dyn) = (meas_T_noisy(:,j) + attack_vec_EKF - z_for_EKF)./sqrt(diag(S));
            Norm_innovation2(:,j - n_dyn) = (meas_T_noisy(:,j) + attack_vec_EKF - z_for_EKF)./sqrt(diag(Rz));
            Residuals_EKF(:,j - n_dyn) = (meas_T_noisy(:,j) + attack_vec_EKF - z_se_EKF);
            Norm_Residuals_EKF(:,j - n_dyn) = (meas_T_noisy(:,j) + attack_vec_EKF - z_se_EKF)./sig(:,j);%./sqrt(abs(omeg(:,j)));     
            Norm_res_paper(:,j - n_dyn) = abs(([theta_WLS_noisy(2:end,j);V_WLS_noisy(:,j)] - estimated_state))./sqrt(diag(P));
        
            
            %%% Predicted and estimated values
            SE_data = [SE_data estimated_state];    % estimated states
            Fo_data = [Fo_data forecasting];        % forecasted state
        end
    end
end

%% plotting results
% plot for bus number 14
set(0,'DefaultAxesFontName','Times New Roman',...
    'DefaultAxesFontSize',12)
fig_plot = 1; %"0" means not plot and "1" means plot
plot_time_WLS = 1/n_dyn_2:24/n_dyn_2:24; % to map the time to 24 hours 
plot_time_EKF = 1/(n_dyn_2-n_dyn):24/(n_dyn_2-n_dyn):24; % to map the time to 24 hours
plot_bus_num = FDIA_state;
disp(' ');
disp(['The results of bus number ',num2str(plot_bus_num),' will be demonstrated']);
disp(' ');
disp(['The qc value has been set to: ',num2str(qc)]);
disp(' ');
V_EKF_plot = SE_data(14:end,:);
del_EKF_plot = SE_data(1:13,:);
V_For_EKF_plot = Fo_data(14:end,:);
del_For_EKF_plot = Fo_data(1:13,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% to plot the results of comparison between EKF and WLS in case needed
if fig_plot == 1

   
    figure('InvertHardcopy','off','Color',[1 1 1]);
    axes('Position',[0.117829454378778 0.107142857142857 0.876803819324264 0.870238095912195]);
    hold on
    plot(V_WLS_true(plot_bus_num,(n_dyn + 1):end),'g');
    plot(V_WLS_noisy(plot_bus_num,(n_dyn + 1):end),'r');
    % plot(plot_time_EKF,V_For_EKF_plot(plot_bus_num,:),'g');
    plot(V_EKF_plot(plot_bus_num,:),'b');
    legend('True-value','SE-WLS','SE-EKF','Location','NorthWest')
    axis tight
    ylabel(['V_{',num2str(plot_bus_num),'} (p.u.)'])
    xlabel('time sample')
    box on

end
