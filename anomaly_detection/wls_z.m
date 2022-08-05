% This has been used to calculate the true measurements

function h = wls_z(num,V,del)
%num = 14;
ybus = ybusppg(num); % Get YBus..
zdata = zdatas(num); % Get Measurement data..
bpq = bbusppg(num); % Get B data..
nbus = max(max(zdata(:,4)),max(zdata(:,5))); % Get number of buses..
type = zdata(:,2); % Type of measurement, Vi - 1, Pi - 2, Qi - 3, Pij - 4, Qij - 5, Iij - 6..
% z = zdata(:,3); % Measuement values..
fbus = zdata(:,4); % From bus..
tbus = zdata(:,5); % To bus..
% Ri = eye(numel(zdata(:,6))); % Measurement Error..
% V = ones(nbus,1); % Initialize the bus voltages..
% del = zeros(nbus,1); % Initialize the bus angles..
% E = [del(2:end); V];   % State Vector..
G = real(ybus);
B = imag(ybus);

vi = find(type == 1); % Index of measurements..
ppi = find(type == 2);
qi = find(type == 3);
pf = find(type == 4);
qf = find(type == 5);

nvi = length(vi); % Number of Voltage measurements..
npi = length(ppi); % Number of Real Power Injection measurements..
nqi = length(qi); % Number of Reactive Power Injection measurements..
npf = length(pf); % Number of Real Power Flow measurements..
nqf = length(qf); % Number of Reactive Power Flow measurements..

% iter = 1;
% tol = 5;

%Measurement Function, h
h1 = V(fbus(vi),1);
h2 = zeros(npi,1);
h3 = zeros(nqi,1);
h4 = zeros(npf,1);
h5 = zeros(nqf,1);

for i = 1:npi
    m = fbus(ppi(i));
    for k = 1:nbus
        h2(i) = h2(i) + V(m)*V(k)*(G(m,k)*cos(del(m)-del(k)) + B(m,k)*sin(del(m)-del(k)));
    end
end

for i = 1:nqi
    m = fbus(qi(i));
    for k = 1:nbus
        h3(i) = h3(i) + V(m)*V(k)*(G(m,k)*sin(del(m)-del(k)) - B(m,k)*cos(del(m)-del(k)));
    end
end

for i = 1:npf
    m = fbus(pf(i));
    n = tbus(pf(i));
    h4(i) = -(V(m)^2)*G(m,n) + V(m)*V(n)*(G(m,n)*cos(del(m)-del(n)) + B(m,n)*sin(del(m)-del(n)));
end

for i = 1:nqf
    m = fbus(qf(i));
    n = tbus(qf(i));
    h5(i) = -(V(m)^2)*(-B(m,n)+bpq(m,n)) + V(m)*V(n)*(G(m,n)*sin(del(m)-del(n)) - B(m,n)*cos(del(m)-del(n)));
end

h = [h1; h2; h3; h4; h5];