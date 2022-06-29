function [B, T, CS, P, U, W, MSE, Q2, Yest, Yhat,  mx, my, XV, YV] = OSCPLSR(X, Y, maxK, orthK)

%  <<  Partial Least Squares Regression prefiltered by Orthogonal signal
%  correction.>>
%
%  Usage:[B, T, CS, P, U, W,  press, Yest, Yhat,  mx, my]  =OSCPLSR(X, Y, maxK, orthK)
%
% X  [L X N]:     INPUT DATA
% Y [L X M];    OUTPUT DATA
% maxk : Number of components to be extracted
% orthK:  Number of orthogonal componet to remove
% << OUTPUT VARIABLES >>
%  B:   Regression Coefficients
%  T:    X scores
%  CS: Y scores
%  P:   X loadings
%  W:  loading weights
%  press: PRESS (LOO varidated)
%  
%  Input data X and Y should be scaled but not centered.
%
% Created in Feb. 2007;     by  Hiroyuki Oya
% Modified in Oct. 2007;     by Hiroyuki Oya

%%
[L,N]=size(X);
[L,M]=size(Y);

% Scaling
% sy=std(Y,[],1);
% sx=std(X,[],1);
% ssx=sx(ones(L,1));
% Y=Y./sy(ones(L,1),:);
% X=X./sx(ones(L,1),:);

% Centering
 mx=mean(X);
 my=mean(Y);
% mmx=repmat(mx,[L 1]);
% X=X-mmx;
% Y=Y-repmat(my,[L,1]);

temX=X;
Yhat=0;

rankX=rank(X);

if orthK>rankX
    disp([  ]);
    error (' Number of PLS component must be less than the rank of INPUT matrix  ....');
end

if orthK>0
    newX=X;
    OW=zeros(N,orthK);
    OP=zeros(N,orthK);
    OT=zeros(L,orthK);

    for n=1:orthK
        rat=1;
        To=ones(L,1);
        u=Y(:,1);
        i=1;
        while rat>=10^-8 && i<150
            % w = X weights 
            w=X'*u/(u'*u);
            w=w/norm(w);
            % Tn = X scores:  Colums of Tn is orthogonal
            Tn=X*w/(w'*w);
            % CS = Y weights:
            C=Y'*Tn/(Tn'*Tn);
            %  u = Y scores:
            u=Y*C/(C'*C);
            % Check convergence
            rat=norm(To-Tn)/norm(Tn);
    %         figure(100);plot(i,rat,'.-');hold on
            To=Tn;
            i=i+1;
        end
        T=To;   % X Scores
        CS=C;   % Y weights
        W=w;    % Weighting vector
        %  P = X loadings: P*C is upper triangular
        P=X'*T/(T'*T);  % X loadings

        % Variables for Orthogonalization
        OW(:,n)=P-(w'*P/(w'*w))*w;      
        r(n)=norm(OW(:,n))/norm(P);
        OWr(:,n)=OW(:,n)/norm(OW(:,n)); 
        OT(:,n)=X*OWr(:,n)/(OWr(:,n)'*OWr(:,n));
        OP(:,n)=X'*OT(:,n)/(OT(:,n)'*OT(:,n));
        % newX : An orthogonal component removed.
        remsig=OT(:,n)*OP(:,n)';
        newX=X-remsig; 
        if norm(newX)/norm(temX)<0.1
            error (' Too many orthogonal componet was removed.....')
        end
        X=newX;
    end

    [B, T, CS, P, U, W,  press, Yhat,  mx, my] = PLSR_HOR(newX, Y, maxK);

elseif orthK==0   
    newX=temX;
    remsig=[];
    [B, T, CS, P, U, W,  press, Yhat,  mx, my] = PLSR_HOR(newX, Y, maxK);
    OT=0;
    OP=0;
    OWr=0;
    r=0;
    XV = var(T*P')/var(newX);
    YV = var(Yhat)/var(Y);
    
    % -----------------    LOO CV   ---------------------- 
    press=0;  sy=0;
    meanx=mean(newX,1);
    meany=mean(Y,1);
    
    for j=1:size(newX,1)
        dd=setdiff([1:size(newX,1)], j);
        [Bi, Ti, CSi, Pi, Ui, Wi,  pressi, Yhati,  mxi, myi]=PLSR_HOR(newX(dd,:), Y(dd), maxK); 
           Yest(j)=(newX(j,:)-mxi)*Bi+myi;                
%         Yest(j)=(newX(j,:)-meanx(j))*Bi+meany;        
        prepress=(Y(j)-Yest(j))^2;
        press=prepress+press;
        presy=(Y(j)-myi)^2;
        sy=sy+presy;
    end
    MSE = press/size(newX,1); % Cross-varidated MSE 
    sy=sum((Y-meany).^2);
    Q2 = 1-(press/sy);             % Q squared = Cross-validated explained variation in Y matrix.
end

figure;subplot(1,2,1);
plot(Y(:,1),Yest,'bd','markerfacecolor','r','markeredgecolor','k','markersize',6); grid on
L1=min(min(Y), min(Yest)) ;  L2=max(max(Y), max(Yest));
xlim([L1-(L2-L1)*0.2  L2+(L2-L1)*0.2]);ylim([L1-(L2-L1)*0.2  L2+(L2-L1)*0.2]);
hold on;
line([L1-(L2-L1)*0.2  L2+(L2-L1)*0.2],[L1-(L2-L1)*0.2  L2+(L2-L1)*0.2])
title(' CV estimation' )

subplot(1,2,2); plot(Y(:,1),Yhat,'.');hold on; grid on
line([L1-(L2-L1)*0.2  L2+(L2-L1)*0.2],[L1-(L2-L1)*0.2  L2+(L2-L1)*0.2])
set(gcf,'position',[890 650 1000 450]); xlabel(' observation'); ylabel('fitted value');
%%

function [B, T, CS, P, U, W,  press, Yhat,  mx, my]=PLSR_HOR(X, Y, maxK)

%% Partial Least Squares Regression 
% X  [L X N]:   INPUT DATA
% Y [L X M];    OUTPUT DATA
% maxk : Number of components to be extracted

% [X]=makedatamatrix(X);
% [Y]=makedatamatrix(Y);

YR=0;
[L,N]=size(X);
[L,M]=size(Y);
%% Centering and scaling if necessary

% Scaling
% sy=std(Y,[],1);
% sx=std(X,[],1);
% Y=Y./sy(ones(L,1),:);
% X=X./sx(ones(L,1),:);

% Centering
mx=mean(X);
my=mean(Y);
mmx=repmat(mx,[L 1]);
X=X-mmx;
Y=Y-repmat(my,[L,1]);
origX=X;
origY=Y;
rankX=rank(X);
%%  PLSR 
if maxK>rankX
    disp('  ');
    error (' Number of PLS component must be less than the rank of INPUT matrix  ....');
end
Yhat=zeros(size(Y));
% main Loop --------------------
for n=1:maxK
    rat=1;
    To=ones(L,1);
    u=Y(:,1);
    i=1;
    while rat>=10^-10 & i<350
        % w = X weights 
        w=X'*u/(u'*u);
        w=w/norm(w);
        % Tn = X scores:  Colums of Tn is orthogonal
        Tn=X*w;
        % CS = Y loading vector:
        C=Y'*Tn/(Tn'*Tn);
        %  u = Y scores:
        u=Y*C/(C'*C);
        % Check convergence
        rat=norm(To-Tn)/norm(Tn);
%         figure(100);plot(i,rat,'.-');hold on
        To=Tn;
        i=i+1;
    end
    T(:,n)=To;     % X Scores
    CS(:,n)=C;    % Y Loadings
    W(:,n)=w;     % Weighting vector
    %  P = X loadings: P*C is upper triangular
    P(:,n)=X'*T(:,n)/(T(:,n)'*T(:,n));    % X loadings
    U(:,n)=u;       % U = Y scores
    %  Deflate X
    X=X-T(:,n)*P(:,n)';    % deflated data matrix X
    TC=T(:,n)*CS(:,n)';  %  Y  explained in the loop
    % Yhat
%     Yhat=Yhat+TC;
%  Deflate Y (not necessary)
    %Y=Y-TC;
%     B=W*((P'*W)\CS');  % PLS regression coefficients
%     YR=Y-Yhat;  %  Y residual

% Sum of squares
ssr=sum(sum(YR.^2));
ssy=sum(sum(origY.^2));
NN(n)=sqrt(ssr)/(L*M);
Rsq(n)=1-ssr/ssy;
press(n)=var(reshape(YR, size(YR,1)*size(YR,2),1));
AIC=[];
%     if M==1
%         AIC(n)=L*log(var(YR))+2*n;
% %           AIC(n)=L*log(var(YR))+log(L)*n*N;
%     else
%         sigma=cov(Y');
%        d=L/(L-(M+n+1));
%        AIC(n)=L*(log(det(sigma))+M)+2*d*(M*n+M*(M+1)/2);
%     end
end     
B=W*((P'*W)\CS');
Yhat=origX*B+my;
%  B=W*C';  % PLS regression coefficients
% B=W*((P'*W)\CS');  % PLS regression coefficients
% YR=Y-X*B;  %  Y residual7
% NN=sqrt(sum(sum(YR.^2))/(L*M));
XR=X;  %  X residual




