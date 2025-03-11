function speakerSet=epdSpeakerRr(auSet)
% epdSpeakerRr: List of RR of each speaker
%
%	Usage:
%		epdSpeakerRr(auSet)
%
%	Example:
%		auDir='waveFile';
%		fprintf('Reading audio files from %s...\n', auDir);
%		auSet=epdAuSetRead(auDir);
%		epdOpt=myEpdOptSet;
%		[recogRate, auSet2]=epdPerfEval(auSet, epdOpt);
%		epdSpeakerRr(auSet2);

%	Roger Jang, 20150328

if nargin<1, selfdemo; return; end

allSpeaker={auSet.speaker};
uniqSpeaker=unique(allSpeaker);
for i=1:length(uniqSpeaker)
	speakerSet(i).name=uniqSpeaker{i};
	index=find(strcmp(allSpeaker, speakerSet(i).name));
	score=[auSet(index).score];
	speakerSet(i).rr=sum(score)/length(index);
end
[junk, index]=sort([speakerSet.rr], 'descend');
speakerSet=speakerSet(index);
rrPlot(speakerSet)

fprintf('Recognition rate for each person:\n');
for i=1:length(speakerSet)
	fprintf('%s: recog. rate=%.2f%%\n', speakerSet(i).name, speakerSet(i).rr*100);
end

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
