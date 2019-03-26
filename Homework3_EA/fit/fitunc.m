function [params,err,hessian] = fitunc(funName,params,freeList,varargin)
%[params,err] = fit(funName,params,freeList,var1,var2,var3,...)
%
%Helpful interface to matlab's 'fminsearch' function.
%
%INPUTS
% 'funName':  function to be optimized.  Must have form err = <funName>(params,var1,var2,...)
% params   :  structure of parameter values for fitted function
%     params.options :  options for fminsearch program (see OPTIMSET)
% freeList :  Cell array containing list of parameter names (strings) to be free in fi
% var<n>   :  extra variables to be sent into fitted function
%
%OUTPUTS
% params   :  structure for best fitting parameters 
% err      :  error value at minimum
% hessian  :  hessian of parameter estimates
%See 'FitDemo.m' for an example.
%
%Modified from Geoff Boynton's fit.m to call fminunc.m instead of
%fminsearch.


%turn free parameters in to 'var'
if isfield(params,'options')
  options = params.options;
else
  options = [];
end


if isempty(freeList)
  freeList = fieldnames(params);
end

vars = params2var(params,freeList);

if ~isfield(params,'shutup')
  disp(sprintf('Fitting "%s" with %d free parameters.',funName,length(vars)));
end

[vars, ~, ~, ~, ~, hessian] = fminunc('fitFunction',vars,options,funName,params,freeList,varargin);

%get final parameters
params = var2params(vars,params,freeList);

%evaluate the function

evalStr = sprintf('err = %s(params',funName);
for i=1:length(varargin)
  evalStr= [evalStr,',varargin{',num2str(i),'}'];
end
evalStr = [evalStr,');'];
eval(evalStr);







