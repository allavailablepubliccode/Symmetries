function particlesim(symmetric)
% set input 'symmetric' to 1 for symmetric case and 0 for invariant case

tpoints = 256;                            	% number of time points
theta   = 8*pi/tpoints:8*pi/tpoints:8*pi;  	% angle

% ellipse constants
%--------------------------------------------------------------------------
a = 1;                                    	% semimajor axis
e = 0.5;                                  	% eccentricity

% equiangular spiral constants
%--------------------------------------------------------------------------
A = 1;
k = 0.2;
eps = 0.1;

% generate radius as function of angle
%--------------------------------------------------------------------------
if symmetric ~= 1
    r = a*(1-e^2)./(1+e*cos(theta));
else
    r = 1./(A*exp(k*theta+eps));
end

xcoord = r.*cos(theta);                     % x-coordinate
ycoord = r.*sin(theta);                   	% y-coordinate

% make figure
%--------------------------------------------------------------------------
figure;
plot(xcoord,ycoord)
drawnow

% model states
%--------------------------------------------------------------------------
x.q    = 1;
x.qdot = 1;

% observation function (to generate timeseries)
%--------------------------------------------------------------------------
g = @(x,v,P) x.q;

% equations of motion
%--------------------------------------------------------------------------
f = @(x,v,P) [x.qdot;...
    ((P.c0/P.c2)*(P.d-P.a)*x.q.^(2-2*P.a)-(P.d+P.a-2)*x.qdot.^2)./(2*x.q)];

% parameters for generalised filtering
%--------------------------------------------------------------------------
E.n  = 4;                              	% embedding dimension
E.d  = 1;                              	% data embedding
E.nN = 8;                              	% number of iterations
E.s  = 1/2;                            	% smoothness of fluctuations

% prior parameters
%--------------------------------------------------------------------------
if symmetric ~= 1
    pE.a  = -3/2;
else
    pE.a  = 2;
end
pE.d  = 0;
pE.c0 = 1;
pE.c2 = 1;

% prior variance
%--------------------------------------------------------------------------
pC.a    = 1;
pC.d    = 1;
pC.c0   = 1;
pC.c2   = 1;

% first level state space model
%--------------------------------------------------------------------------
DEM.M(1).E  = E;                        % filtering parameters
DEM.M(1).x  = x;                        % initial states
DEM.M(1).f  = f;                        % equations of motion
DEM.M(1).g  = g;                        % observation mapping
DEM.M(1).pE = pE;                       % model parameters
DEM.M(1).pC = diag(spm_vec(pC))*16;     % variance
DEM.M(1).V  = exp(12);                  % precision of observation noise
DEM.M(1).W  = exp(12);                  % precision of state noise

% second level causes or exogenous forcing term
%--------------------------------------------------------------------------
DEM.M(2).v  = 0;                        % initial causes
DEM.M(2).V  = exp(16);                  % precision of exogenous causes

% data and known input
%--------------------------------------------------------------------------
DEM.Y = zscore(r);
DEM.U = zeros(1,tpoints);

% Inversion using generalised filtering
%==========================================================================
LAP = spm_DEM(DEM);

% use Bayesian model reduction to test different hypotheses
%==========================================================================
model{1} = 'invariant';
model{2} = 'symmetric';

% apply precise shrinkage priors
%--------------------------------------------------------------------------
PC{1} = pC;                             % full model (scale invariant)
PC{2} = pC; PC{2}.d = 0;                % reduced model (scale symmetric)

%  evaluate the evidence for these new models or prior constraints
%--------------------------------------------------------------------------
qE    = LAP.qP.P{1};
qC    = LAP.qP.C;
pE    = LAP.M(1).pE;
pC    = LAP.M(1).pC;
for m = 1:numel(PC)
    rC     = diag(spm_vec(PC{m}));
    F(m,1) = spm_log_evidence(qE,qC,pE,pC,pE,rC);
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