using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(NeuralTrainerEvolution_Follow))]
public class NeuralTrainerEvolution_FollowEditor
    : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        if (!Application.isPlaying)
            return;

        var self = target as NeuralTrainerEvolution_Follow;

        if (GUILayout.Button("Loop"))
            self.Loop();

        self.cycleTime = EditorGUILayout.Slider("Cycle Time", self.cycleTime, 0.5f, 20.0f);

        self.copyRate = EditorGUILayout.Slider("Copy Rate", self.copyRate, 0.0f, 1.0f);
        self.mutateRate = EditorGUILayout.Slider("Mutate Rate", self.mutateRate, 0.0f, 1.0f);
        self.evolveRate = EditorGUILayout.Slider("Evolve Rate", self.evolveRate, 0.0f, 1.0f);

        self.hyper = EditorGUILayout.Toggle("Hyper", self.hyper);
        self.hyperSpeed = EditorGUILayout.IntSlider("Hyper Speed", self.hyperSpeed, 1, 100);

        self.trainingPhase = EditorGUILayout.IntSlider("Training Phase", self.trainingPhase, 0, 5);

        self.showDebugLines = EditorGUILayout.Toggle("Show Debug Lines", self.showDebugLines);
    }
}
#endif

public class NeuralTrainerEvolution_Follow
    : MonoBehaviour
{
    public Transform target;
    public NeuralBurst source;

    [System.NonSerialized] public float cycleTime = 5.0f;

    [System.NonSerialized] public bool hyper = false;
    [System.NonSerialized] public int hyperSpeed = 5;

    [System.NonSerialized] public float copyRate = 0.0f;
    [System.NonSerialized] public float mutateRate = 0.001f;
    [System.NonSerialized] public float evolveRate = 0.10f;

    [System.NonSerialized] public int trainingPhase = 0;

    [System.NonSerialized] public bool showDebugLines = false;

    public class TrainingResult
    {
        public NeuralBurst trainee;
        public float score;
    };

    private List<TrainingResult> results;
    private List<NeuralBurst> trainees;

    private float bestScore;
    private NeuralBurst bestTrainee;

    public void OnEnable()
    {
        trainees = new List<NeuralBurst>();

        for (int i = 0; i < 64; ++i)
        {
            var obj = GameObject.Instantiate(source.gameObject);

            obj.name = string.Format("{0}-{1}", source.gameObject.name, i);

            var neural = obj.GetComponent<NeuralBurst>();

            neural.Automatic = false;
            neural.Configure(6, 64000, 4);
            neural.Randomize();

            trainees.Add(neural);
        }

        source.gameObject.SetActive(false);

        Loop();
    }

    public void Loop()
    {
        StartCoroutine(RunTrainingLoop());
    }

    public void ClearResults()
    {
        if (results == null)
            results = new List<TrainingResult>();
        results.Clear();

        for (int i = 0; i < trainees.Count; ++i)
            results.Add(new TrainingResult() { trainee = trainees[i], score = 0.0f, });
    }

    private int CompareResults(TrainingResult lhs, TrainingResult rhs)
    {
        if (lhs.score < rhs.score)
            return -1;
        if (lhs.score > rhs.score)
            return +1;
        return 0;
    }

    public IEnumerator RunTrainingLoop()
    {
        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): running...");

        var cycle = 0;

        bestScore = float.MaxValue;
        bestTrainee = (NeuralBurst)null;

        var stepJobs = new List<JobHandle>(trainees.Count);
        var mutateJobs = new List<JobHandle>(trainees.Count);
        var evolveJobs = new List<JobHandle>(trainees.Count);

        while (true)
        {
            ClearResults();

            var worldOffset = Vector3.zero;

            if (trainingPhase >= 5)
                worldOffset = Random.onUnitSphere * Random.Range(10.0f, 20.0f);

            for (int i = 0; i < trainees.Count; ++i)
            {
                var trainee = trainees[i];
                var result = results[i];

                var direction = Random.onUnitSphere;
                direction.y = 0.0f;
                direction = direction.normalized;

                var position = new Vector3(
                    Random.Range(-10.0f, 10.0f),
                    Random.Range(-10.0f, 10.0f),
                    Random.Range(-10.0f, 10.0f));

                trainee.transform.position = position + worldOffset;
                trainee.transform.rotation = Quaternion.LookRotation(Random.onUnitSphere);

                trainee.GetComponent<Renderer>().material.color = trainee.GenerateColor();
            }

            switch (trainingPhase)
            {
                case 0:
                    target.transform.position = Vector3.zero;
                    break;

                case 1:
                    {
                        var position = Random.onUnitSphere * 5.0f;
                        position.y = 0.0f;
                        target.transform.position = position;
                    }
                    break;

                case 2:
                    target.transform.position = Random.onUnitSphere * 5.0f;
                    break;

                case 3:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f);
                    break;

                case 4:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f);
                    break;

                case 5:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f) + worldOffset;
                    break;
            }

            var counter = 0.0f;

            while (counter < cycleTime)
            {
                if (hyper)
                {
                    Physics.autoSimulation = false;
                    Time.timeScale = 1.0f * (float)hyperSpeed;

                    TickPhase(Time.fixedDeltaTime);

                    for (int h = 0; h < hyperSpeed; ++h)
                    {
                        for (int i = 0; i < trainees.Count; ++i)
                        {
                            var trainee = trainees[i];
                            var result = results[i];
                            var dt = Time.fixedDeltaTime;
                            var job = PrepareAndQueueJobStep(trainee, dt);

                           stepJobs.Add(job.Schedule());

                            if ((i % 8) == 0)
                                JobHandle.ScheduleBatchedJobs();
                        }

                        for (int i = 0; i < trainees.Count; ++i)
                        {
                            var trainee = trainees[i];
                            var handle = stepJobs[i];

                            handle.Complete();

                            AfterTrainingJob(trainee, Time.fixedDeltaTime);
                        }

                        stepJobs.Clear();

                        Physics.Simulate(Time.fixedDeltaTime);
                    }

                    counter += Time.fixedDeltaTime * Time.timeScale;

                    yield return null;
                }
                else
                {
                    Physics.autoSimulation = true;
                    Time.timeScale = 1.0f;

                    TickPhase(Time.fixedDeltaTime);

                    UnityEngine.Profiling.Profiler.BeginSample("QueueTrainingJobs");

                    for (int i = 0; i < trainees.Count; ++i)
                    {
                        var trainee = trainees[i];
                        var result = results[i];
                        var job = PrepareAndQueueJobStep(trainee, Time.fixedDeltaTime);

                        stepJobs.Add(job.Schedule());

                        if ((i % 8) == 0)
                            JobHandle.ScheduleBatchedJobs();
                    }

                    UnityEngine.Profiling.Profiler.EndSample();

                    UnityEngine.Profiling.Profiler.BeginSample("AfterTrainingJobs");

                    for (int i = 0; i < trainees.Count; ++i)
                    {
                        var trainee = trainees[i];
                        var handle = stepJobs[i];

                        handle.Complete();

                        AfterTrainingJob(trainee, Time.fixedDeltaTime);
                    }

                    stepJobs.Clear();

                    UnityEngine.Profiling.Profiler.EndSample();

                    counter += Time.fixedDeltaTime;

                    yield return new WaitForFixedUpdate();
                }
            }

            for (int i = 0; i < trainees.Count; ++i)
            {
                var trainee = trainees[i];
                var result = results[i];

                CalculateResult(trainee, result);
            }

            results.Sort(CompareResults);

            var winner = results[0];
            if (winner.score < bestScore)
            {
                Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): cycle {0} +++ new winner: {1}: score: {2}", cycle, winner.trainee.name, winner.score);

                bestScore = winner.score;
                bestTrainee = winner.trainee;

                bestScore += 0.1f;
            }
            else
            {
                Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): cycle {0} === old winner: {1}: score: {2}", cycle, bestTrainee.name, bestScore);

                bestScore += 1.0f;
            }

            for (int i = 0; i < trainees.Count; ++i)
            {
                var seed = (uint)(Time.frameCount * (i + 1));

                if (trainees[i] == bestTrainee)
                    continue;

                var jobMutate = trainees[i].QueueJobMutate(seed, mutateRate);
                var jobEvolve = trainees[i].QueueJobEvolve(seed, bestTrainee, evolveRate);

                var jobHandleMutate = jobMutate.Schedule();
                var jobHandleEvolve = jobEvolve.Schedule(jobHandleMutate);

                mutateJobs.Add(jobHandleMutate);
                evolveJobs.Add(jobHandleEvolve);

                if ((i % 8) == 0)
                    JobHandle.ScheduleBatchedJobs();
            }

            // this shouldnt be needed but getting dependency error
            for (int i = 0; i < mutateJobs.Count; ++i)
                mutateJobs[i].Complete();
            mutateJobs.Clear();

            for (int i = 0; i < evolveJobs.Count; ++i)
                evolveJobs[i].Complete();
            evolveJobs.Clear();

            for (int i = 0; i < trainees.Count; ++i)
                trainees[i].cachedRigidbody.isKinematic = true;
            yield return null;
            for (int i = 0; i < trainees.Count; ++i)
                trainees[i].cachedRigidbody.isKinematic = false;

            cycle++;

            yield return null;
        }
    }

    public void TickPhase(float dt)
    {
        if (trainingPhase >= 4)
        {
            target.transform.position += new Vector3(
                Mathf.Sin(Time.time * 0.1f) * 10.0f * dt,
                Mathf.Cos(Time.time * 0.2f) * 0.5f * dt,
                Mathf.Sin(Time.time * 0.3f) * 10.0f * dt
            );

        }
    }

    public NeuralBurst.StepJob PrepareAndQueueJobStep(NeuralBurst trainee, float dt)
    {
        var traineeBody = trainee.cachedRigidbody;

        trainee.state0[0] = target.transform.position.x - trainee.transform.position.x;
        trainee.state0[1] = target.transform.position.y - trainee.transform.position.y;
        trainee.state0[2] = target.transform.position.z - trainee.transform.position.z;

        trainee.state0[3] = traineeBody.velocity.x;
        trainee.state0[4] = traineeBody.velocity.y;
        trainee.state0[5] = traineeBody.velocity.z;

        return trainee.QueueJobStep();
    }

    public void AfterTrainingJob(NeuralBurst trainee, float dt)
    {
        var traineeBody = trainee.cachedRigidbody;

        var brake = traineeBody.velocity * -trainee.state2[3];
        var force = new Vector3(
            trainee.state2[0],
            trainee.state2[1],
            trainee.state2[2]
        ) * 1000.0f;

        //if (float.IsNaN(force.x)) force.x = 0.0f;
        //if (float.IsNaN(force.y)) force.y = 0.0f;
        //if (float.IsNaN(force.z)) force.z = 0.0f;

        var apply = (brake + force) * dt;

        traineeBody.AddForce(apply, ForceMode.Acceleration);

        if (showDebugLines)
        {
            //Debug.DrawLine(new Vector3(trainee.state0[0], trainee.state0[1], trainee.state0[2]), new Vector3(trainee.state0[3], trainee.state0[4], trainee.state0[5]), new Color(trainee.state2[0] * 2.0f, trainee.state2[1] * 2.0f, trainee.state2[2] * 2.0f));
            //Debug.DrawLine(new Vector3(trainee.state0[0], trainee.state0[1], trainee.state0[2]), new Vector3(trainee.state0[6], trainee.state0[7], trainee.state0[8]), Color.white);

            var accuracy = Vector3.Dot(force.normalized, (target.position - traineeBody.position).normalized);
            Debug.DrawLine(trainee.transform.position, trainee.transform.position + force * 0.01f, Color.Lerp(Color.white, Color.red, accuracy));
        }
    }

    public void CalculateResult(NeuralBurst trainee, TrainingResult result)
    {
        var ofs = trainee.transform.position - target.transform.position;
        var lsq = ofs.sqrMagnitude;
        var len = 0.0f;

        if (lsq > 0.001f)
            len = ofs.magnitude;

        result.score = len;
    }
}
