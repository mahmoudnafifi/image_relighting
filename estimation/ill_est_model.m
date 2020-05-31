classdef ill_est_model
    properties
        features %training features
        temps %training temperatures
        K %K value for KNN
        encoder %PCA object
        
    end
    methods
        function feature = encode(obj,hist,I) %encode RGB-histogram feature
            feature =  obj.encoder.encode(hist);
            
            
        end
        
        function hist = RGB_UVhist(obj,I)  %generate RGB-histogram
            I = im2double(I);
            h= sqrt(max(size(obj.encoder.weights,1),...
                size(obj.encoder.weights,2))/3);
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
        end
        
        function [est_ill] = estimate_ill (obj,I,feature, sigma) %Image correction
            I = im2double(I);
            if nargin == 2
                feature = obj.encode(obj.RGB_UVhist(I),I);
                sigma = 0.45;
            elseif nargin == 3
                sigma = 0.45;
            end
            [dH,idH] = pdist2(obj.features,feature,...
                'euclidean','Smallest',obj.K);
            
            weightsH = exp(-((dH).^2)/(2*sigma^2));
            weightsH = weightsH/sum(weightsH);
            
            est_ill = sum(weightsH .* obj.temps(idH,:),1);
        end
    end
end
