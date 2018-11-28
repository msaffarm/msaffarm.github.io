function ImageSeg_MGDM(image,J)

% ImageSeg_MFDM takes an RGB image and applies MGDM segmentation
% J is the number of distinct colors in the input image.
% $Author: Mansour Saffar Mehrjardi

%% make dataset based on image
    image=im2double(image);
    [rows,cols,~]=size(image);
    X=reshape(image,rows*cols,3);

%% set treshold and tolerance and maxium epochs
    rand('seed',0);
    tolerance=1.e-3;
%     max_iteration=1000;
    max_iteration = 25;
%% EM algorithm
N=size(X,1);%number of samples or data dimension!!
%create symmetric positive definite matrice for  Sigma
sigma_old=zeros(3,3,J);
a=randn(3,3);
b=a'*a+3*eye(3);
for j=1:J
    sigma_old(:,:,j)=b;
end
%
mu_old=rand(3,1,J);%first mu
P_j_old=(1/J)*ones(J,1);%channel probabilities (Uniform Distribution)
%%
iteration_num=1;

while(1)
  P_j_xk = zeros(N,J);
  for m=1:J
    P_j_xk(:,m) = mvnpdf( X, mu_old(:,:,m)', sigma_old(:,:,m) ) * P_j_old(m);
  end
  p_xk = sum(P_j_xk,2);
  P_j_xk = P_j_xk ./ repmat( p_xk,1,J);
 %M step
 mu_new=zeros(3,1,J);
 sigma_new=zeros(3,3,J);
 P_j_new=zeros(J,1);
 for m=1:J
    mu_new(:,:,m)=(sum(repmat(P_j_xk(:,m),1,3).*X))'./(sum(P_j_xk(:,m)));
    b=X-repmat(mu_new(:,:,m)',N,1);
    c=repmat(sqrt(P_j_xk(:,m)),1,3).*b;
    sigma_new(:,:,m)= (c'*c)./(sum(P_j_xk(:,m))) + 1.e-4*eye(3,3);% to avoid singular matrix;
    P_j_new(m,:)=mean(P_j_xk(:,m));
 end
 mu_d = sum(sqrt((mu_old-mu_new).^2),1);
 sig2_d=sum(sum(sqrt((sigma_new-sigma_old)),1),2);
 P_j_d=norm(P_j_old-P_j_new);
 iteration_num=iteration_num+1;
 %% check if tolerance is reached!!
 mu_d_reached=0;
 if(mu_d < tolerance)
     mu_d_reached=1;
 end
 sig2_d_reached=0;
 if(sig2_d < tolerance)
     sig2_d_reached=1;
 end
 pjd_reached=1;
 if(P_j_d<tolerance)
     pjd_reached=1;
 end
 if(iteration_num > max_iteration ||( mu_d_reached && sig2_d_reached  && pjd_reached  ) )
     break;
 end
 mu_old=mu_new;
 sigma_old=sigma_new;
 P_j_old=P_j_new;
end
%% assign each pixel to its gaussian pdf based on channel probability
RGBColors=[255 0 0 204 127 255 128;255 102 204 0 0 128 128;0 204 0 102 255 0 128];%% RGB color map (including 7 distinct colors)
datamat=zeros(rows*cols,J);
for h=1:J
    datamat(:,h)=P_j_new(h).*mvnpdf(X,mu_new(:,:,h)',sigma_new(:,:,h));
end
[~,max_iter]=max(datamat,[],2);
% make RGB color channels
R=RGBColors(1,max_iter);
G=RGBColors(2,max_iter);
B=RGBColors(3,max_iter);
segmented_img=cat(2,R,G,B);
final_img1=reshape(segmented_img,rows,cols,3)./255;


%% Display result
subplot(1,2,1), imshow(final_img1,[]);
title('Segmented image according to l=argmax w*N(xi;\mu)')
subplot(1,2,2), imshow(image,[]);
title('Original Image')
end