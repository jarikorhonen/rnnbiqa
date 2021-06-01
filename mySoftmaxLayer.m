% Custom Softmax layer

classdef mySoftmaxLayer < nnet.layer.Layer
    % Example custom PReLU layer.

    properties (Learnable)
        % Layer learnable parameters
            
    end
    
    methods
        function layer = mySoftmaxLayer(name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % for 2-D image input with numChannels channels and specifies 
            % the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Custom softmax layer";
        end
        
%         function dLdY = backwardLoss(layer, Y, T)
%             % loss = backwardLoss(layer, Y, T)
%             
%             N = size(Y,4);
%             Y = squeeze(Y);
%             T = squeeze(T);
%     
%             dLdY = -T./Y+(1-T)./(1-Y);
%         end        
%         
%         function loss = forwardLoss(layer, Y, T)
%             % loss = forwardLoss(layer, Y, T) returns the weighted cross
%             % entropy loss between the predictions Y and the training
%             % targets T.
% 
%             N = size(Y,4);
%             Y = squeeze(Y);
%             T = squeeze(T);
%     
%             loss = -sum(T.*log(Y))/N;
%         end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            XTemp = X(1,1,:,:);
            Z = exp(XTemp)./sum(exp(XTemp));
            Z(1,1,:,:) = Z;
        end
    end
end