% Line Data for B-Bus (Shunt Admittance)Formation.
% Praviraj PG, Oct-2009, IIT Roorkee

function bbus = bbusppg(num)     % Returns B-bus..

linedata = linedatas(num);
fb = linedata(:,1);
tb = linedata(:,2);
b = linedata(:,5);
nbus = max(max(fb),max(tb));    % no. of buses...
nbranch = length(fb);           % no. of branches...
bbus = zeros(nbus,nbus);

 for k=1:nbranch
     bbus(fb(k),tb(k)) = b(k);
     bbus(tb(k),fb(k)) = bbus(fb(k),tb(k));
 end