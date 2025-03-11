function [recogRate, auSet, time]=epdPerfEval(auSet, epdOpt)
% epdPerfEval: EPD (endpoint detection) performance evaluation
%
%	Usage:
%		[recogRate, auSet, time]=epdPerfEval(auSet);
%		[recogRate, auSet, time]=epdPerfEval(auSet, epdOpt);
%
%	Description:
%		recogRate=epdPerfEval(auSet) returns the recognition rate of myEpd over the given auSet
%		See the following example.
%
%	Example:
%		auDir='waveFile';
%		fprintf('Reading audio files from %s...\n', auDir);
%		auSet=epdAuSetRead(auDir);
%		epdOpt=myEpdOptSet;
%		[recogRate, auSet2, time]=epdPerfEval(auSet, epdOpt);
%		fprintf('Overall recognition rate = %.2f%%.\n', recogRate*100);
%		fprintf('Time = %g sec\n', time);

%	Roger Jang, 20150328

if nargin<1, selfdemo; return; end
if nargin<2, epdOpt=myEpdOptSet; end

% ====== End-point detection for each file
myTic=tic;
auNum=length(auSet);
for i=1:auNum
%	fprintf('%d/%d: %s\n', i, auNum, auSet(i).path);
	auSet(i).epByComputer=myEpd(auSet(i), epdOpt);
	auSet(i).score=epdScore(auSet(i).epByComputer, auSet(i).epByHuman);
end

% ====== Overall recognition rate
recogRate=sum([auSet.score])/auNum;
time=toc(myTic);

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
