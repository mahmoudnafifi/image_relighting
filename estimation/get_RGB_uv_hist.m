%
% Copyright (c) 2018-present, Mahmoud Afifi
% York University, Canada
% mafifi@eecs.yorku.ca | m.3afifi@gmail.com
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
% All rights reserved.
%
% Please cite the following work if this program is used:
% -------------------------------------------------------
% Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
% "When color constancy goes wrong: Correcting improperly white-balanced
% images", CVPR 2019.
%
% Mahmoud Afifi and Michael S. Brown. Sensor Independent Illumination 
% Estimation for DNN Models. In BMVC, 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%


function hist = get_RGB_uv_hist(I,h)
if nargin == 1
    h  =60;
end
I = imresize(im2double(I),[256,256]);
eps= 6.4/h;
I=(reshape(I,[],3));
A=[-3.2:eps:3.19];
hist=zeros(size(A,2),size(A,2),3);
i_ind=I(:,1)~=0 & I(:,2)~=0 & I(:,3)~=0;
I=I(i_ind,:);
Iy=sqrt(I(:,1).^2+I(:,2).^2+I(:,3).^2);
for i = 1 : 3
    r = setdiff([1,2,3],i);
    Iu=log((I(:,i))./(I(:,r(1))));
    Iv=log((I(:,i))./(I(:,r(2))));
    diff_u=abs(Iu-A);
    diff_v=abs(Iv-A);
    diff_u=(reshape((reshape(diff_u,[],1)<=eps/2),[],size(A,2)));
    diff_v=(reshape((reshape(diff_v,[],1)<=eps/2),[],size(A,2)));
    hist(:,:,i)=(Iy.*double(diff_u))'*double(diff_v);
    hist(:,:,i)=sqrt(hist(:,:,i)/sum(sum(hist(:,:,i))));
end
hist = imresize(hist,[h h],'bilinear');