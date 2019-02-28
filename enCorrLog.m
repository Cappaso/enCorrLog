function [B,A] = enCorrLog(X,Y,rho1,rho2,lam)
%%% Augmented logistic regression for multilabel data
%      p(y|x) \proportion_to exp{x*B*y + 0.5*y'*A*y};
%  i.e., p(y|x) \proportion_to exp{ \sum_i yi*<x,Bi> + \sum_{i<j}A_ij*yi*yj };
% input: X n by (p-1)+1, where p-1 is feature dimensionality, and the added
%        1 is for bias parameter
%        Y n by m, 0-1 matrix. where m is label number;
%        rho is weight parameter
% output: B m by (p-1), the model
%         A m by m symmetric matrix, with diag(A) = 0, A_ij models the
%         correlation between y_i and y_j

star_t = tic;
m = size(Y,2);
[n,p] = size(X);
xB0 = zeros(m,p) ;        % initialize B
yB = xB0;
xA0 = zeros(m,m); xA0 =xA0 - diag(diag(xA0)); % initialize A
yA = xA0;

maxIter = 500;  %max iteration
t = 1/((sum(X(:).^2))/n)*30;          %step size
if nargin==4; lam = 1; end % sparse level
tol = 1e-3;
tc = 1;

Y_ = (-1).^Y;
X = full(X);

for ii = 1:maxIter
    
    %%%%%% calculate objective and gradient
    % objetive on yB and yA
    eta = X*yB'; % n by m
    xi = Y*yA;  % n by m
    T = eta + xi; % n by m
    fv(ii) = 1/n*sum(sum(log(1+exp(Y_.*T)))) +...
        rho1*(sum(sum((yB(:,1:end-1).^2))) + lam*sum(sum(abs(yB(:,1:end-1))))) + rho2*(sum(sum(yA(:).^2)) + lam*sum(abs(yA(:))));
    
    % gradient on yB and yA
    Xi = Y_./(1+exp(-Y_.*T)); % n by m
    H_B=Xi'*X/n; 
    H_B(:,1:end-1) = H_B(:,1:end-1) + 2*rho1*yB(:,1:end-1);
    
    H_A = Xi'*Y/n + 2*rho2*yA; 
    H_A = H_A - diag(diag(H_A));
    H_A = H_A + H_A';
    
    % generalized gradient descent updating
    xB = yB - t*H_B;
    xBtmp1 = abs(xB);
    xBtmp1 = max(0,xBtmp1 - t*rho1*lam);
    xBtmp2 = sign(xB);
    xB = xBtmp1.*xBtmp2;
    
    xA = yA - t*H_A;
    xAtmp1 = abs(xA);
    xAtmp1 = max(0,xAtmp1 - t*rho2*lam);
    xAtmp2 = sign(xA);
    xA = xAtmp1.*xAtmp2;
    
    tc_ = (1+sqrt(1+4*tc^2))/2;
    if ii>20 && abs((fv(ii)-fv(ii-1))/fv(ii))<tol
        conv_t = toc(star_t);
        disp(['converged with ' num2str(conv_t) ' seconds']);
        B = xB;
        A = xA;
        return
    end
    yB = xB + (tc-1)/(tc_)*(xB-xB0);
    yA = xA + (tc-1)/(tc_)*(xA-xA0);
    tc = tc_;
    xB0 = xB;
    xA0 = xA;

end
B = xB;
A = xA;
if ii > maxIter
    conv_t = toc(star_t);
    disp(['not converged within ' num2str(conv_t) ' seconds']);
end