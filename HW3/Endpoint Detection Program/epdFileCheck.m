function epdFileCheck(auSet, speaker)
% epdFileCheck: Check each file after EPD
%
%	Usage:
%		epdFileCheck(auSet)
%		epdFileCheck(auSet, speaker)
%
%	Description:
%		epdFileCheck(auSet) check each audio (from the badly performed ones) in auSet by displaying its endpoints
%			auSet: a structure array of an audio set (See the following example as how to obtain the array)
%
%	Example:
%		auDir='waveFile';
%		fprintf('Reading audio files from %s...\n', auDir);
%		auSet=epdAuSetRead(auDir);
%		epdOpt=myEpdOptSet;
%		[recogRate, auSet2]=epdPerfEval(auSet, epdOpt);
%		epdFileCheck(auSet2);

%	Roger Jang, 20150328

if nargin<1; selfdemo; return; end
% ====== Take the data from a specific speaker
if nargin>1
	auSet=auSet(strcmp(speaker, {auSet.speaker}));
	if isempty(auSet), error('Cannot find files from the given speaker %s!', speaker); end
end
if isempty(auSet), error('Given auSet is empty!'); end

for i=1:length(auSet)
	auSet(i).absDiff=mean(abs(auSet(i).epByHuman-auSet(i).epByComputer))/auSet(i).fs;
end
[junk, index]=sort([auSet.absDiff], 'descend');

for i=1:length(auSet)
	j=index(i);
	fprintf('%d/%d: %s (score=%g, absDiff=%g)\n', i, length(auSet), auSet(j).file, auSet(j).score, auSet(j).absDiff);
	epdPlot(auSet(j));
	fprintf('Hit any key to continue...\n'); pause;
end

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
