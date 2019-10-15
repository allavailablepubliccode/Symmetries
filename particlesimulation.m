function particlesimulation(symmetric)
% set input 'symmetric' = 1 to run scale symmetric case for a particle
% that is acted upon by a force that varies inversely as the cube of
% position, resulting in an equiangular spiral trajectory

% set input 'symmetric' = 0 to run scale invariant case for a particle
% that is acted upon by a force that varies inversely as the square of
% position, resulting in an elliptical trajectory

rng(0,'twister')

if symmetric == 1
    disp('ground truth - scale symmetric')
else
    disp('ground truth - scale invariant')
end

disp('runtime ~ few mins')

% ellipse constants
%--------------------------------------------------------------------------
a = 1;                                    	% semimajor axis
e = 0.5;                                  	% eccentricity

% equiangular spiral constants
%--------------------------------------------------------------------------
A = 1;
k = 0.2;
eps = 0.1;

% model states
%--------------------------------------------------------------------------
x.xcoord    = 1;                            % x position
x.ycoord    = 1;                            % y position
x.xcoorddot = 1;                            % x velocity
x.ycoorddot = 1;                            % y velocity

% observation function
%--------------------------------------------------------------------------
g = @(x,v,P) [x.xcoord;x.ycoord];

% equation of motion (Euler-Lagrange via 5th order Lagrangian)
%--------------------------------------------------------------------------
f = @(x,v,P) [x.xcoorddot;...
    x.ycoorddot;...
    (x.xcoord.^(-1-2*P.a)*(P.c0*...
    (-P.a+P.d)*x.xcoord.^4-P.c2*(-2+P.a+P.d)*x.xcoord.^(2+2*P.a)*...
    x.xcoorddot.^2-2*P.c3*(-3+2*P.a+P.d)*x.xcoord.^(1+3*P.a)*...
    x.xcoorddot.^3-3*P.c4*(-4+3*P.a+P.d)*x.xcoord.^(4*P.a)*...
    x.xcoorddot.^4))./(2*(P.c2*x.xcoord.^2+3*P.c3*x.xcoord.^(1+P.a).*...
    x.xcoorddot+6*P.c4*x.xcoord.^(2*P.a).*x.xcoorddot.^2));...
    (x.ycoord...
    .^(-1-2*P.a)*(P.c0*(-P.a+P.d)*x.ycoord.^4-P.c2*(-2+P.a+P.d)*...
    x.ycoord.^(2+2*P.a)*x.ycoorddot.^2-2*P.c3*(-3+2*P.a+P.d)*x.ycoord...
    .^(1+3*P.a)*x.ycoorddot.^3-3*P.c4*(-4+3*P.a+P.d)*x.ycoord.^(4*P.a)...
    *x.ycoorddot.^4))./(2*(P.c2*x.ycoord.^2+3*P.c3*x.ycoord.^(1+P.a).*...
    x.ycoorddot+6*P.c4*x.ycoord.^(2*P.a).*x.ycoorddot.^2))];

% generate ground truth data
%--------------------------------------------------------------------------
tpoints = 100;                            	% number of time points
theta   = 8*pi/tpoints:8*pi/tpoints:8*pi;  	% angle

if symmetric ~= 1
    r = a*(1-e^2)./(1+e*cos(theta));        % ellipse
else
    r = 1./(A*exp(k*theta+eps));            % equiangular spiral
end

Y = [r.*cos(theta);r.*sin(theta)];          % data to model

% parameters for generalised filtering
%--------------------------------------------------------------------------
E.n  = 4;                                   % embedding dimension
E.d  = 1;                                   % data embedding
E.nN = 8;                                   % number of iterations
E.s  = 1/2;                                 % smoothness of fluctuations

% prior parameters
%--------------------------------------------------------------------------
% alpha
if symmetric ~= 1
    pE.a  = 3/2;        % inverse square force law (Kepler exponent)
else
    pE.a  = 2;          % inverse cube force law
end

pE.d  = 0;              % delta (deviation from symmetry)

pE.c0 = 1/64;           % coefficients - n.b. must be non-zero for
pE.c2 = 1/64;           % integrator to initialize in absense of external
pE.c3 = 1/64;           % driving input
pE.c4 = 1/64;

% prior variance
%--------------------------------------------------------------------------
pC.a  = 0;
pC.d  = 16;
pC.c0 = 16;
pC.c2 = 16;
pC.c3 = 16;
pC.c4 = 16;

% first level state space model
%--------------------------------------------------------------------------
DEM.M(1).E  = E;                      	% filtering parameters
DEM.M(1).x  = x;                      	% initial states
DEM.M(1).f  = f;                      	% equations of motion
DEM.M(1).g  = g;                       	% observation mapping
DEM.M(1).pE = pE;                      	% model parameters
DEM.M(1).pC = diag(spm_vec(pC));    	% variance
DEM.M(1).V  = exp(9);               	% precision of observation noise
DEM.M(1).W  = exp(9);                	% precision of state noise

% n.b. precisions of observation and state noise set to exp(6) for fMRI
% and calcium imaging data due to higher noise assumption

% second level causes or exogenous forcing term
%--------------------------------------------------------------------------
DEM.M(2).v  = 0;                      	% initial causes
DEM.M(2).V  = exp(9);                 	% precision of exogenous causes

% data and known input
%--------------------------------------------------------------------------
DEM.Y = Y;                              % data to model
DEM.U = zeros(1,tpoints);               % zero driving input

% n.b. driving input set to rand(1,tpoints) for fMRI and calcium imaging
% data to model neural fluctuations

% Inversion using generalised filtering
%==========================================================================
LAP = spm_DEM(DEM);

% use Bayesian model reduction to test different hypotheses
%==========================================================================
model{1} = '1) symmetric';
model{2} = '2) invariant';

% apply precise shrinkage priors
%--------------------------------------------------------------------------
PC{1} = pC;  PC{1}.d = 0;           % reduced model (scale symmetric)
PC{2} = pC;                         % full model (scale invariant)

%  evaluate the evidence for these new models or prior constraints
%--------------------------------------------------------------------------
qE    = LAP.qP.P{1};
qC    = LAP.qP.C;
pE    = LAP.M(1).pE;
pC    = LAP.M(1).pC;
for m = 1:numel(PC)
    rC     = diag(spm_vec(PC{m}));
    [F(m,1) sE(m,1)] = spm_log_evidence(qE,qC,pE,pC,pE,rC);
end

% report marginal log likelihood or evidence
%--------------------------------------------------------------------------
F = F - min(F);

spm_figure('GetWin','Model Comparison');clf;
subplot(2,2,1), bar(F,'c')
title('Log evidence','FontSize',16)
xlabel(model), axis square, box off

subplot(2,2,2), bar(spm_softmax(F(:)),'c')
title('Probability','FontSize',16)
xlabel(model), axis square, box off

% run model forward with posterior densities
%--------------------------------------------------------------------------
tpoints = 200;

% extract velocities for Noether charge
g = @(x,v,P) [x.xcoord;x.ycoord;x.xcoorddot;x.ycoorddot];

% furnish parameters with posterior densities
if symmetric==1
    P.a  = sE(1).a;
    P.d  = sE(1).d;
    P.c0 = sE(1).c0;
    P.c2 = sE(1).c2;
    P.c3 = sE(1).c3;
    P.c4 = sE(1).c4;
else
    P.a  = sE(2).a;
    P.d  = sE(2).d;
    P.c0 = sE(2).c0;
    P.c2 = sE(2).c2;
    P.c3 = sE(2).c3;
    P.c4 = sE(2).c4;
end

% turn down noise to see pure functional forms

M(1).E  = E;                      	% filtering parameters
M(1).x  = x;                      	% initial states
M(1).f  = f;                      	% equations of motion
M(1).g  = g;                       	% observation mapping
M(1).pE = P;                      	% model parameters
M(1).V  = exp(30);               	% precision of observation noise
M(1).W  = exp(30);                	% precision of state noise
M(2).v  = 0;                    	% initial causes
M(2).V  = exp(30);                 	% precision of exogenous causes

U = zeros(1,tpoints);               % external perturbation

DEMgen = spm_DEM_generate(M,U,P);   % generate data

% Noether charge
Noeth = P.a.*P.c0.*(1:tpoints).*full(DEMgen.Y(2,:)).^(-P.a+P.d)...
    +P.c4.*full(DEMgen.Y(2,:)).^(-4+3.*P.a+P.d).*...
    full(DEMgen.Y(4,:)).^3.*(4.*full(DEMgen.Y(2,:))-3.*...
    P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c3.*...
    full(DEMgen.Y(2,:)).^(-3+2.*P.a+P.d).*...
    full(DEMgen.Y(4,:)).^2.*(3.*full(DEMgen.Y(2,:))-...
    2.*P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c2.*...
    full(DEMgen.Y(2,:)).^(-2+P.a+P.d).*...
    full(DEMgen.Y(4,:)).*(2.*full(DEMgen.Y(2,:))-P.a.*...
    (1:tpoints).*full(DEMgen.Y(4,:)));

figure;
plot(Noeth(101:end))       
title('Noether charge, particle')

if symmetric == 1
    ylim([-0.365 -0.245])
else
    ylim([-0.27 -0.15])
end

% % Monkey
% %--------------------------------------------------------------------------
% clear P
% P.a  = 0.886;
% P.d  = -5.264e-06;
% P.c0 = 0.005;
% P.c2 = 0.035;
% P.c3 = -0.115;
% P.c4 = 0.111;
% 
% M(1).pE = P;
% 
% tpoints = 400;
% 
% U = zeros(1,tpoints);             
% 
% DEMgen = spm_DEM_generate(M,U,P);  
% 
% Noeth = P.a.*P.c0.*(1:tpoints).*full(DEMgen.Y(2,:)).^(-P.a+P.d)...
%     +P.c4.*full(DEMgen.Y(2,:)).^(-4+3.*P.a+P.d).*...
%     full(DEMgen.Y(4,:)).^3.*(4.*full(DEMgen.Y(2,:))-3.*...
%     P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c3.*...
%     full(DEMgen.Y(2,:)).^(-3+2.*P.a+P.d).*...
%     full(DEMgen.Y(4,:)).^2.*(3.*full(DEMgen.Y(2,:))-...
%     2.*P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c2.*...
%     full(DEMgen.Y(2,:)).^(-2+P.a+P.d).*...
%     full(DEMgen.Y(4,:)).*(2.*full(DEMgen.Y(2,:))-P.a.*...
%     (1:tpoints).*full(DEMgen.Y(4,:)));
% 
% figure;
% plot(Noeth(101:end))    
% title('Noether charge, Monkey')
% 
% ylim([0 0.1])
% 
% 
% % Mice
% %--------------------------------------------------------------------------
% clear P
% P.a  = 1.855;
% P.d  = 1.055e-05;
% P.c0 = 0.016;
% P.c2 = 0.499;
% P.c3 = -0.495;
% P.c4 = 0.072;
% 
% M(1).pE = P;
% 
% tpoints = 200;
% 
% U = zeros(1,tpoints);              
% 
% DEMgen = spm_DEM_generate(M,U,P);  
% 
% Noeth = P.a.*P.c0.*(1:tpoints).*full(DEMgen.Y(2,:)).^(-P.a+P.d)...
%     +P.c4.*full(DEMgen.Y(2,:)).^(-4+3.*P.a+P.d).*...
%     full(DEMgen.Y(4,:)).^3.*(4.*full(DEMgen.Y(2,:))-3.*...
%     P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c3.*...
%     full(DEMgen.Y(2,:)).^(-3+2.*P.a+P.d).*...
%     full(DEMgen.Y(4,:)).^2.*(3.*full(DEMgen.Y(2,:))-...
%     2.*P.a.*(1:tpoints).*full(DEMgen.Y(4,:)))+P.c2.*...
%     full(DEMgen.Y(2,:)).^(-2+P.a+P.d).*...
%     full(DEMgen.Y(4,:)).*(2.*full(DEMgen.Y(2,:))-P.a.*...
%     (1:tpoints).*full(DEMgen.Y(4,:)));
% 
% figure;
% plot(Noeth(101:end))        
% title('Noether charge, Mice')
% 
% ylim([-2.3 -2.1])
