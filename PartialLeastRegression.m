function  [B, T, W,C, P, u] =PartialLeastRegression(nX, nY, maxK)

%   <<<    Partial Least Squares Regression    >>>
%                  Based on NIPALS algolithm
%                               PLS-1/2
%
% nX  [L x N]:     INPUT DATA
% nY  [L x M];    OUTPUT DATA
% maxk : Number of components to be extracted
%
% /******   Outputs   ******/
% B: overall regression coefficients
% W : PLS weights (transformed)
% T: PLS scores (orthogonal)
% P: X-loadings 
% u: PLS Weights (orthonormal)
% C: Y-loadings
% varE : variance in X explained for each component

% Created by H.Oya 

%% --------------------------------------------------------------------  %%
[L,N]=size(nX);
[L,M]=size(nY);

rankX=rank(nX);

if maxK>rankX
    disp('  ');
    error (' Number of PLS component must be less than the rank of INPUT matrix  ....');
end
Yhat=zeros(size(nY));
vy1=sum(nY.^2);
%%  ///******   main NIPALS Loop   *******///  
for n=1:maxK
    YX=nY'*nX;
   % PLS Weights: u
    u(:,n)=YX(1,:)'/norm(YX(1,:));      
     if size(nY,2)>1 
        uold=u(:,n)+1;
        while norm(u(:,n)-uold)>10^-10
            uold=u(:,n);
            tu=YX'*YX*u(:,n);
            u(:,n)=tu/norm(tu);
        end
     end      
    % X score
    T(:,n)=nX*u(:,n);
    % Y loadings
    C(:,n)=nY'*T(:,n)/(T(:,n)'*T(:,n));
    % X loadings
    P(:,n)=nX'*T(:,n)/(T(:,n)'*T(:,n));
    % Yhat
    Yhat=Yhat+T(:,n)*C(:,n)';
    % Deflation of X and Y
    nX=nX-T(:,n)*P(:,n)';
    nY=nY-T(:,n)*C(:,n)';
    % var explained in Y
    vy=sum(nY.^2);
    Rsq(n)=1-(vy/vy1);
end

%%  Transformed weights W 
W=u*inv(P'*u);
W(:,1)=u(:,1);

%%  Overall regression coefficients...
% note :  P'*u is upper diagonal with 1s on main diagonal
B=u*((P'*u)\C');

%% Compute variance explained...
% tem2=sum(sum(orignX.^2));
% for n=1:maxK
%        tem1(n)=sum(sum((T(:,n)*P(:,n)').^2));
% end
% varE=tem1./tem2;
%         

%% /***********************************/
function [newX, sx, mx] = centernormalize(X)
% Centering and Scaling the matrix X

[L,N]=size(X);
% Scaling ...G
  sx=std(X,[],1);
  newX=X./sx(ones(L,1),:);

% Centering ...
mx=mean(X,1);
mmx=repmat(mx,[L 1]);
newX=X-mmx;

