function y = map_maxprod_01(eta,A)
%%% y = map_maxprod_01(eta,A)
%      p(y) \proportion_to exp{eta*y + 0.5*y'*A*y};
% max-product algorithm for binary MAP estimation on 0-1 ising model
%   input: eta, m by 1 vector, linear coefficients
%          A, m by m symetric matrix, quadratic coefficients

m = length(eta);
y = zeros(m,1);
Miter = 50;
Diageta = sparse(diag(eta));
gamma_0 = sparse(zeros(m,m));
gamma_1 = gamma_0;
gamma_1 = sparse(gamma_1 + diag(eta));
loc = (0:m-1)*m + (1:m);

for itr = 1:Miter
%     for ii = 1:m
% %         for jj = 1:m
% %             r0 = sum(gamma_0(:,ii)) - gamma_0(jj,ii);
% %             r1 = sum(gamma_1(:,ii)) - gamma_1(jj,ii);
% %             ggamma_0(ii,jj) = max(r0, r1);
% %             ggamma_1(ii,jj) = max(r0, r1 + A(ii,jj));
% %         end
%         rr0 = sum(gamma_0(:,ii)) - gamma_0(:,ii)';
%         rr1 = sum(gamma_1(:,ii)) - gamma_1(:,ii)';
%         ggamma_0(ii,:) = max(rr0,rr1);
%         ggamma_1(ii,:) = max(rr0,rr1 + A(ii,:));
%     end
    rr0 = bsxfun(@minus,sum(gamma_0,1)', gamma_0');
    rr1 = bsxfun(@minus,sum(gamma_1,1)', gamma_1');
    ggamma_0 = max(rr0,rr1);
    ggamma_1 = max(rr0,rr1 + A);
    
    gamma_0 = ggamma_0; gamma_1 = ggamma_1;
    Mgamma = max(gamma_0,gamma_1);
    gamma_0 = gamma_0 - Mgamma; gamma_1 = gamma_1 - Mgamma;
    gamma_1(loc) = 0;
    gamma_1 = gamma_1 + Diageta;    
end
gamma_0 = sum(gamma_0,1);
gamma_1 = sum(gamma_1,1);
loc = gamma_1>gamma_0;
y(loc) = 1;
