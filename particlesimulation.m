function particlesimulation(symmetric)
% set input 'symmetric' = 1 to run scale symmetric case for a particle
% that is acted upon by a force that varies inversely as the cube of
% position, resulting in an equiangular spiral trajectory

% set input 'symmetric' = 0 to run scale invariant case for a particle
% that is acted upon by a force that varies inversely as the square of
% position, resulting in an elliptical trajectory

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

% equation of motion (5th order Lagrangian)
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
pC.d  = 1;
pC.c0 = 1;
pC.c2 = 1;
pC.c3 = 1;
pC.c4 = 1;

% first level state space model
%--------------------------------------------------------------------------
DEM.M(1).E  = E;                      	% filtering parameters
DEM.M(1).x  = x;                      	% initial states
DEM.M(1).f  = f;                      	% equations of motion
DEM.M(1).g  = g;                       	% observation mapping
DEM.M(1).pE = pE;                      	% model parameters
DEM.M(1).pC = diag(spm_vec(pC))*16;    	% variance
DEM.M(1).V  = exp(9);               	% precision of observation noise
DEM.M(1).W  = exp(9);                	% precision of state noise

% n.b. precisions of observation and state noise set to exp(6) for fMRI
% data due to higher noise assumption

% second level causes or exogenous forcing term
%--------------------------------------------------------------------------
DEM.M(2).v  = 0;                      	% initial causes
DEM.M(2).V  = exp(9);                 	% precision of exogenous causes

% data and known input
%--------------------------------------------------------------------------
DEM.Y = Y;                              % data to model
DEM.U = zeros(1,tpoints);               % zero driving input

% n.b. driving input set to rand(1,tpoints) for fMRI data to model
% afferent neural fluctuations

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

% Mathematica code to plot Noether charge using posterior expectations
% sE in Matlab code above
%--------------------------------------------------------------------------

% (*Scale symmetric posterior expectations*)
% a = 2;
% d = -2.2345*10^(-6);
% c0 = 0.0151;
% c2 = 0.0847;
% c3 = -0.0565;
% c4 = 0.0563;
% 
% (*Scale invariant posterior expectations*)
% (*a =3/2;
% d=-0.0205;
% c0=0.0070;
% c2=0.0183;
% c3=-0.0174;
% c4=0.0548;*)
% 
% (*Velocity*)
% q'[t] = D[q[t], t];
% 
% (*5th order Lagrangian*)
% 
% Lagr = q[t]^(d - a) (c0 + c2 q[t]^(2 a - 2) q'[t]^2 + 
%      c3 q[t]^(3 a - 3) q'[t]^3 + c4 q[t]^(4 a - 4) q'[t]^4);
% 
% (*Noether charge*)
% 
% Noet = FullSimplify[(Lagr - q'[t] D[Lagr, q'[t]]) a t + 
%     D[Lagr, q'[t]] q[t]];
% 
% (*Euler Lagrange*)
% 
% EL = NDSolve[{D[Lagr, q[t]] == D[D[Lagr, q'[t]], t], q[0] == 1, 
%     q'[0] == 1}, q, {t, 0, 100}];
% 
% (*plot Noether charge*)
% Plot[Evaluate[Noet /. EL], {t, 0, 296}, 
%  PlotRange -> {{0, 100}, {0.1, 0.25}}]
% 
% (*fMRI posterior expectations*)
% (*a =0.8857;
% d=-1.9747*10^(-6);
% c0=0.0054;
% c2=0.0348;
% c3=-0.1152;
% c4=0.1107;*)
% 
% (*Calcium imaging posterior expectations*)
% (*a =1.2698;
% d=1.0552*10^(-5);
% c0=0.0164;
% c2=0.4990;
% c3=-0.4945;
% c4=0.0720;*)
