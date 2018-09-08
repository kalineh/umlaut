﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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

        if (GUILayout.Button("Run"))
            self.Run();
        if (GUILayout.Button("Loop"))
            self.Loop();
    }
}
#endif

public class NeuralTrainerEvolution_Follow
    : MonoBehaviour
{
    public Transform target;
    public List<Neural> trainees;

    public class TrainingResult
    {
        public Neural trainee;
        public bool done;
        public float score;
    };

    private List<TrainingResult> results;

    public void OnEnable()
    {
        foreach (var trainee in trainees)
        {
            trainee.Automatic = false;
            trainee.Reset();
            trainee.Configure(6, 12, 3);
            trainee.Randomize();
        }

        Run();
    }

    public void Run()
    {
        StartCoroutine(RunTrainingIteration());
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
            results.Add(new TrainingResult() { trainee = trainees[i], done = false, score = 0.0f, });
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
        while (true)
            yield return RunTrainingIteration();
    }

    public IEnumerator RunTrainingIteration()
    {
        var steps = 100;

        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): running...");

        ClearResults();

        for (int i = 0; i < trainees.Count; ++i)
        {
            var trainee = trainees[i];
            var result = results[i];

            StartCoroutine(RunTraining(trainee, result, steps));
        }

        for (int i = 0; i < steps; ++i)
            yield return null;

        // wait for complete
        yield return null;
        yield return null;
        yield return null;
        yield return null;

        results.Sort(CompareResults);

        var winner = results[0];

        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): winner: {0}: score: {1}", winner.trainee.name, winner.score);
        Debug.LogFormat("");

        for (int i = 0; i < trainees.Count; ++i)
            trainees[i].LerpTowards(winner.trainee, 0.1f);
    }

    public IEnumerator RunTraining(Neural trainee, TrainingResult results, int steps)
    {
        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTraining(): {0}", trainee.name);

        var position = trainee.transform.position;
        var rotation = trainee.transform.rotation;

        var input = new float[6];
        var traineeBody = trainee.GetComponent<Rigidbody>();

        for (int i = 0; i < steps; ++i)
        {
            input[0] = trainee.transform.position.x;
            input[1] = trainee.transform.position.y;
            input[2] = trainee.transform.position.z;

            input[3] = target.transform.position.x;
            input[4] = target.transform.position.y;
            input[5] = target.transform.position.z;

            trainee.SetInputs(input);
            trainee.Step();

            yield return null;

            var force = new Vector3(
                trainee.state2[0],
                trainee.state2[1],
                trainee.state2[2]
            );

            traineeBody.AddForce(force / Time.fixedDeltaTime, ForceMode.Acceleration);
        }

        var ofs = trainee.transform.position - target.transform.position;
        var lsq = ofs.sqrMagnitude;

        results.score = lsq;
        results.done = true;

        trainee.transform.position = position;
        trainee.transform.rotation = rotation;

        traineeBody.velocity = Vector3.zero;
        traineeBody.angularVelocity = Vector3.zero;
        traineeBody.isKinematic = true;

        yield return null;

        traineeBody.isKinematic = false;

        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTraining(): {0}: score {1}", trainee.name, results.score);

        yield break;
    }
}