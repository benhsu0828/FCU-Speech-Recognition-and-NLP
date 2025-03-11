function epdPlot(au)
% epdPlot: Display audio signals and their end-points
%
%	Usage:
%		epdPlot(au)
%
%	Descruiption:
%		This function is used in epdFileCheck.m.
%
%	Example:
%		auFile='waveFile/123456_Zhehui/1b_11023_15336.wav';
%		au=myAudioRead(auFile);
%		epdOpt=myEpdOptSet;
%		au.epByComputer=myEpd(au, epdOpt);
%		au.epByHuman=[11023, 15336];
%		epdPlot(au);

%	Roger Jang, 20150328

if nargin<1; selfdemo; return; end

wave=au.signal;
fs=au.fs;
epByHuman=au.epByHuman;
epByComputer=au.epByComputer;

subplot(2,1,1);
plot(wave);
axis([-inf inf -1 1]);
ylim=get(gca, 'ylim');
line(epByHuman(1)*[1 1], ylim, 'color', 'r', 'linewidth', 2);
line(epByHuman(2)*[1 1], ylim, 'color', 'r', 'linewidth', 2);
line(epByComputer(1)*[1 1], ylim, 'color', 'k', 'linewidth', 2);
line(epByComputer(2)*[1 1], ylim, 'color', 'k', 'linewidth', 2);
set(gca, 'xlim', [-inf inf]);
score=epdScore(epByComputer, epByHuman, fs);
title(['score=', num2str(score), ' (Red: by human, black: by computer)']);
subplot(2,1,2);
cutted=wave(epByHuman(1):epByHuman(2));
plot(cutted);
axis([-inf inf -1 1]);
sound(wave, fs);
% ====== Display buttons
U.wave=wave; U.fs=fs;
U.voicedY=[];
if ~isempty(epByComputer)
	U.voicedY=U.wave(epByComputer(1):epByComputer(end));
end
set(gcf, 'userData', U);
uicontrol('string', 'Play all', 'callback', 'U=get(gcf, ''userData''); sound(U.wave, U.fs);', 'position', [20, 20, 100, 20]);
uicontrol('string', 'Play detected', 'callback', 'U=get(gcf, ''userData''); sound(U.voicedY, U.fs);', 'position', [150, 20, 80, 20]);

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
