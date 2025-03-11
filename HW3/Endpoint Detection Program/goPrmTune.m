% This is an example of exhaustive search to find the optimal parameter value for EPD

auDir='waveFile';
%auDir='/users/jang/temp/epdTrain';
fprintf('Reading wave files from "%s"...\n', auDir);
auSet=epdAuSetRead(auDir);

epdOpt=myEpdOptSet;
volumeRatio=linspace(eps, 0.3, 20);
recogRate=0*volumeRatio;
for i=1:length(volumeRatio)
	epdOpt.volumeRatio=volumeRatio(i);
	recogRate(i)=epdPerfEval(auSet, epdOpt);
	fprintf('%d/%d: Volume ratio = %g, Recog. rate = %.2f%%.\n', i, length(volumeRatio), epdOpt.volumeRatio, recogRate(i)*100);
end

plot(volumeRatio, recogRate*100, '.-');
[maxRecogRate, index]=max(recogRate);
fprintf('Best volume ratio = %g, best RR = %g%%\n', volumeRatio(index), maxRecogRate*100);
line(volumeRatio(index), maxRecogRate*100, 'color', 'r', 'marker', 'o');
xlabel('volumeRatio'); ylabel('Recog. Rate (%)');
