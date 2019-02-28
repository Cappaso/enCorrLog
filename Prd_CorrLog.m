function Y = Prd_CorrLog(X,B,A)
n = size(X,1);
m = size(B,1);
Y = zeros(n,m);
print_cnt = 20;
print_step = ceil(n/print_cnt);

for in = 1:n
    Y(in,:) = map_maxprod_01(B*(X(in,:)'),A);
    
%     if mod(in,print_step)==0
%         fprintf('--Test data %d of %d\n',in,n)
%     end
end