classdef Polyhedral < dagnn.ElementWise
    properties
        useShortCircuit = true
        leak = 0
        opts = {}
        muCache = [];
        scoreCache = [];
        updateMuCache = 1;
       % mode = '';
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            %outputs{1} = cat(3, inputs{1}, inputs{1}) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %derInputs{1} = derOutputs{1}(:,:,1:size(derOutputs{1},3)/2,:) + derOutputs{1}(:,:,size(derOutputs{1},3)/2 + 1:end,:);
            %derParams = {} ;
        end
        
        function forwardAdvanced(obj, layer)
            if ~obj.useShortCircuit || ~obj.net.conserveMemory
                forwardAdvanced@dagnn.Layer(obj, layer) ;
                return ;
            end
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            inputs = net.vars(in);
            scores = gather(inputs(1).value);
            labels = gather(inputs(2).value);
            
            noClasses = size(net.params(end).value, 1);
            featureSize = size(scores,3);
            batchSize = size(scores,4);  
            
            if (strcmp(net.mode, 'test'))
                mu = obj.muCache;
                output = zeros(1, 1, featureSize * 2, batchSize);
                for indImage = 1:batchSize
                    output(1, 1, :, indImage) = cat(3, scores(1, 1, :, indImage) - mu(1, 1, :), abs(scores(1, 1, :, indImage) - mu(1, 1, :)));
                end
            elseif (obj.updateMuCache == 0)
                mu = obj.muCache;
                
                output = zeros(1, 1, featureSize * 2, batchSize);
                for indImage = 1:batchSize
                    output(1, 1, :, indImage) = cat(3, scores(1, 1, :, indImage) - mu(1, 1, :), abs(scores(1, 1, :, indImage) - mu(1, 1, :)));
                end
            else   
                mu = zeros(1, 1, featureSize);
                if  (noClasses==1)
                    freq = sum(labels==1);
                    for indImage = 1:batchSize,
                        if labels(indImage)==1,
                            mu(1, 1, :) = mu(1, 1, :) + (scores(1,1,:,indImage));
                        end
                    end
                    mu = mu/freq;
                    if freq <= 1
                        if ~isempty(obj.muCache)
                            mu(1, 1, :) = obj.muCache(1, 1, :);
                        else
                            mu(1, 1, :) = 0;
                        end
                    end
                     if freq >= 2
                         obj.muCache = mu;
                     end
                else
                    for indImage = 1:batchSize
                        mu(1, 1, :) = mu(1, 1, :) + (scores(1,1,:,indImage)/batchSize);
                    end
                    
                    obj.muCache = mu;
                    
                end
                                      
               
                                
                output = zeros(1, 1, featureSize * 2, batchSize);
                for indImage = 1:batchSize
                    output(1, 1, :, indImage) = cat(3, scores(1, 1, :, indImage) - mu(1, 1, :), abs(scores(1, 1, :, indImage) - mu(1, 1, :)));
                end
                           
            end
            
            obj.scoreCache = output;
            
            net.vars(out).value = gpuArray(single(output));
            
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if ~net.vars(v).precious & net.numPendingVarRefs(v) == 0
                    net.vars(v).value = [] ;
                end
            end
        end
        
        function backwardAdvanced(obj, layer)
            if ~obj.useShortCircuit || ~obj.net.conserveMemory
                backwardAdvanced@dagnn.Layer(obj, layer) ;
                return ;
            end
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            
            if isempty(net.vars(out).der), return ; end
            
            derInput = net.vars(out).der(:,:,1:size(net.vars(out).der,3)/2,:) + sign(obj.scoreCache(:, :, size(obj.scoreCache, 3)/2+1:end, :)) .* net.vars(out).der(:,:,size(net.vars(out).der,3)/2+1:end,:);
            
            if ~net.vars(out).precious
                net.vars(out).der = [] ;
                net.vars(out).value = [] ;
            end
            
            if net.numPendingVarRefs(in(1)) == 0
                net.vars(in(1)).der = derInput ;
            else
                net.vars(in(1)).der = net.vars(in(1)).der + derInput ;
            end
            net.numPendingVarRefs(in(1)) = net.numPendingVarRefs(in(1)) + 1 ;
            
            net.vars(in(2)).der = [];
            %net.numPendingVarRefs(in(2)) = net.numPendingVarRefs(in(2)) + 1 ;
        end
        
        function obj = Polyhedral(varargin)
        %    obj.mode = varargin{2};
            obj.load(varargin) ;
        end
    end
end
