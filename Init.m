function Init

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Field Dimensions - x and y maximum (in meters)
run=3;
iteration=5;
PopNum=30;
VelocityLimit=3.83;
xRank=13.33;  
RepLimit = 100;    % repository limitation number . we can not exceed this number of record in repository
ObjNum=2;
R = 6;
C1 = [1.3 1.5 1.8 1.9 2];      % C1 nd C2 range
Weight = [0.6 0.7 0.8 0.9 1];          % W range
%Energy Model (all values in Joules)
%Initial Energy 
Eo=8;
NodeNum=100;
PlotSizeX=100;
PlotSizeY=100;
X=PlotSizeX*rand(1,NodeNum);
Y=PlotSizeY*rand(1,NodeNum);


E=Eo*ones(1,NodeNum); %Initil Energy
sender.x=0;
sender.y=0;
%x and y Coordinates of the Sink
sink.x=PlotSizeX/2;
sink.y=PlotSizeY/2;





% % %Number of Nodes in the field
% % n=300;
% % %Optimal Election Probability of a node
% % %to become cluster head
% % p=0.2;


%Eelec=Etx=Erx
ETX=50*0.000000001;
ERX=50*0.000000001;
%Transmit Amplifier types
Efs=10*0.000000000001;
Emp=0.0013*0.000000000001;
%Data Aggregation Energy
EDA=5*0.000000001;
%maximum number of rounds
rmax=5;

%%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%
%Computation of do
do=sqrt(Efs/Emp);
MutualDominance=1;
MobilityRate=2;%movement rate of sink
Step=50; %step of modeling
BsTrackX=[sink.x];
BsTrackY=[sink.y];  % save the track of x and y coordinate of sink

energy_values=[];
fitness=[];
index_energy=1;
index_fitness=1;

for w=1:Step
    TrackSize=size(BsTrackX);
    [sink.x sink.y]=NodeMovement(BsTrackX(TrackSize(2)),BsTrackY(TrackSize(2)),PlotSizeX,PlotSizeY,MobilityRate);
    BsTrackX=[BsTrackX sink.x];
    BsTrackY=[BsTrackY sink.y];
    SenderNum=ceil(rand()*7);
    for jj=1:SenderNum
        I=ceil(NodeNum*rand());
        while (E(I)<=0)
            I=ceil(NodeNum*rand());
        end
        sender.x=X(I);
        sender.y=Y(I);
        for ii=1:run
           [OutputParticleAC OutputParticleVal OutputFitness OutputRepSize Energy]=PSO(iteration,X,Y,E,sink,sender,xRank,RepLimit,PopNum,C1(3),Weight(4),ETX,EDA,Emp,NodeNum,PlotSizeX,ObjNum,do,VelocityLimit,R,Efs,MutualDominance,I,MobilityRate);
        end
        E=Energy;
        energy_values(index_energy,1:100)=E;
        fitness(index_fitness,1)=OutputFitness(1,1);
        fitness(index_fitness,2)=OutputFitness(1,2);
        fitness(index_fitness+1,1)=OutputFitness(2,1);
        fitness(index_fitness+1,2)=OutputFitness(2,2);
        index_fitness=index_fitness+2;
        index_energy=index_energy+1;
        
    end
    disp(['end of step ',num2str(w),' ']);
end
h=figure()
plot(energy_values)
saveas(h,'energy.fig')
close(h)

h=figure()
plot(fitness)
axis([0 650 -70 -30])
saveas(h,'fitness.fig')

disp('The optimal energy of clusters : ')
a=size(energy_values);
energy_values(a(1),:)

disp('The optimal fintness value or no. of clusters : ')
a=size(fitness);
disp(-fitness(a(1),2));

end