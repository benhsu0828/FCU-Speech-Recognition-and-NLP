function score=epdScore(endPointByComputer, endPointByHuman, fs, timeDiff)
% epdScore: Compute the recognition score of EPD
%
%	Usage:
%		score=epdScore(endPointByComputer, endPointByHuman, fs, timeDiff)
%
%	Example:
%		auFile='waveFile/123456_Zhehui/1b_11023_15336.wav';
%		au=myAudioRead(auFile);
%		epdOpt=myEpdOptSet;
%		au.epByComputer=myEpd(au, epdOpt);
%		au.epByHuman=[11023, 15336];
%		epdPlot(au);
%		score=epdScore(au.epByComputer, au.epByHuman, au.fs);
%		fprintf('score=%f\n', score);

%	Roger Jang, 20060304

if nargin<1, selfdemo; return; end
if nargin<3, fs=16000; end
if nargin<4, timeDiff=0.0625; end	% 1000/16000=0.0625¬í

if length(endPointByComputer)~=2, score=0; return; end
if length(endPointByHuman)~=2, score=0; return; end

score=mean(abs(endPointByComputer-endPointByHuman)<fs*timeDiff*[1 1]);

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
