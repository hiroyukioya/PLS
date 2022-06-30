function [recon, score,eivalue,eivector,avg] = PCAanal_v2(data, delp, recdata)
% PCA denoising ...
%
% PCA analysis of data ( NOTE:  data = [ Observation X Dimension]   )
% projvec: projection to the eigenvectors
% recon:  reconstruction using (selp) principal axes
% delp=[ 1 2 3...] delete these eigen modes for reconstruction 
% 
%   X = [observation x feature dimension]
%   X = SUV' from singular value decomposition
%   Score (score) = XV = USV'V = US 
%   Reconstructed data (recon) = XVV' (=USV')
%-------------------------------------------------------%

[a,b]=size(data);

% Find mean
avg=mean(data,1);  
avgi = mean(recdata,1);

% Centering
cdata = (data - avg(ones(a,1),:));

% Perform SVD 
[U,S,V] = svds(cdata./sqrt(a-1),b);  
dp = size(V,2);

% Eigen vectors 
eivector = V;

% Eigen values 
eivalue=diag(S).^2;

% Score 
score = cdata*V;

% Reconstruction excluding "delp" PCs......
V(:,delp)=0;
recon = recdata*V*V';%- repmat(avgi, size(recdata,1),1) + repmat(avg, size(recdata,1),1);

su=sum(eivalue);
csum=cumsum(eivalue)./su; 


%-------------------------------------------------------%
% figure;plot(csum,'-*'),ylim([0 1]),title('Cumulative percentage of total variance')
% figure;pleivalue),title('Scree plot')